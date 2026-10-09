# The graph-stride x DSpark defect

Status: **root-caused at the interface, worked around by configuration.** The
deployed profile (`profiles/fp8/`) runs static `k=7` with `FULL_DECODE_ONLY`
graphs and **no** `SPEC_TOKENS_SCHEDULE`. Do not re-add the dynamic schedule
without reading this file.

Host these numbers came from: 2x Intel Arc Pro B70, TP2, XPU, image
`vllm-llmscaler:icr26.35`, target `Qwen3.8-27B-FP8`, drafter
`qwen3.8-27b-dflash-drafter-fp8-b70`. Correctness figures are 5 prompts x 3 reps
per boot, 2 boots per arm unless stated.

## The defect

`FULL_DECODE_ONLY` is correct with DSpark **iff every captured uniform decode
stride equals `1 + num_speculative_tokens`** (the scheduler's own
`decode_query_len`). A FULL decode graph captured at any other stride replays
another position's logits.

The DSpark speculator always drafts the **scalar** `num_speculative_tokens`
(`v1/worker/gpu/spec_decode/speculator.py:77` sets `num_speculative_steps` from
it; `dspark/speculator.py:117` loops `for i in range(n_spec)`). The rejection
sampler and the input buffers are sized from that same scalar. But the capture
loop takes its strides from the per-batch-size schedule
(`v1/worker/gpu/cudagraph_utils.py:216-218`), and
`num_speculative_tokens_per_batch_size` appears **nowhere** in the spec_decode
worker path. The scheduler can draft *n* tokens while the engine captures and
replays a graph built for *m*.

The signature is `'duct'`, empty strings, and CJK fragments — with acceptance
counters **identical** to a healthy run. Verification still runs; only the
rejected-draft fallback reads the wrong buffer.

**Consequence for anyone operating this stack: throughput and acceptance metrics
cannot detect this. Only a correctness probe can.** That is why
`scripts/bench/correctness.sh` runs on every deploy and why its expected probe
count is a hard gate rather than a rate.

## Measured (graphs ON)

| schedule | k | captured stride(s) | correct | C1 tok/s | conc=8 tok/s |
|---|---|---|---|---|---|
| none | 7 | 8 | **30/30** | 150.41 | 165.63 |
| none | 6 | 7 | **30/30** | ~101 | 170.9-175.7 |
| none | 4 | 5 | **6-7/15 CORRUPT** | ~105 | 176.27 |
| none | 1 | 2 | **30/30** | 42.30 | — |
| `[[1,1,7],[2,64,7]]` | 7 | 8 | **30/30** | 149.78 | — |
| `[[1,1,4],[2,64,4]]` | 4 | 5 | **14/30 CORRUPT** | 105.52 | — |
| `[[1,1,7],[2,64,4]]` | 7/4 | 8 and 5 | **13/30 CORRUPT** | 151.42 | 175.09 |
| none, graphs OFF | 4 | — | **30/30** | 20.77 | 102.58 |
| none, graphs OFF | 7 | — | **15/15** | 21.80 | 107.68 |

## What was falsified on the way (recorded so it is not re-litigated)

Each of these cost a boot or a run. All six were plausible, and two were my own
load-bearing theories.

1. **Multi-shape capture is the trigger** — falsified. A single captured shape
   still corrupts (static k=4: 6/15).
2. **Async scheduling causes it** — falsified (13/30 corrupt with async off).
3. **The dynamic path per se causes it** — falsified. A dynamic schedule whose
   strides are all 7 is 30/30 correct. The *stride value* matters, not the
   schedule mechanism.
4. **Capture-time stride inconsistency is the mechanism** — falsified by static
   k=4, which has manager stride 5, `decode_query_lens=[5]`, no schedule at all,
   and is still 6/15 corrupt *and graphed*. This is why the patch below is
   honestly a workaround and not a fix.
5. **The capture-grid rule ("the stride must divide the capture size")** —
   falsified by k=6: stride 7 divides no capture size any better than k=4's
   stride 5, yet k=6 is 30/30 correct across four boots.
6. **The drafter is just broken at k!=7** — falsified by static k=4 with graphs
   **OFF**: 30/30 correct. So the corruption *requires* a replayed graph. This
   keeps the "cudagraphs x spec decode" framing correct and means production's k
   is not intrinsically fragile, only the graph interaction.

**What survives:** the fault is k-specific **and** graph-dependent. k=1, 6, 7
replay correctly; k=4 does not. No parity, divisibility, or drafter-alone rule
explains that pattern. The mechanism inside the graph path was not found. Treat
"root-caused" as "root-caused to the interface", not to a line of code.

## The workaround, and the rule it implies

Static `k=7`, `cudagraph_mode=FULL_DECODE_ONLY`, **no schedule**. Then every
captured stride is `1+7=8` by construction, which is the safe shape.

## The patch (validated, NOT deployed, not shipped in this repo)

A patch that makes the unsafe shape unreachable was written and causally
validated: it drops capture for any stride inconsistent with the scheduler's, and
uncovered shapes then get `cg_mode=NONE` from `dispatch` and run eager, so no
wrong graph can ever be replayed. The gate was opt-in
(`VLLM_XPU_SPEC_CG_STRICT_STRIDE=1`; unset = upstream behaviour byte-for-byte).
The causal test hit 6/6 boots as pre-registered: dynamic 7/4 went 7/15 -> 30/30,
static k=4 went 6/15 -> 30/30 (losing its only graph, so 38.78 tok/s), and a
no-op control moved 174.64 -> 171.25, within noise.

It is **not deployed and deliberately not included in this repo**, for two
reasons:

- patched-dynamic measured conc=8 **106.39 vs static k=7's 165.63** (-35.8%),
  because the k=4 shape falls back to eager; and
- a bind-mounted `.py` can silently stop applying after an image rebuild — a
  config that degrades silently is worse than one that is correct by
  configuration.

Static k=7 is faster, needs no mount, and cannot be broken by a rebuild. If you
re-derive the patch, keep the two properties that made it safe: read the gate via
`__import__("os").environ` (an undeclared `envs.VLLM_XPU_*` raises
`AttributeError`, and an undeclared env var is exactly how a bind-mounted file
silently stops mattering), and pre-format log messages rather than passing
non-hashable args into a `warning_once` that could raise mid-capture.

## Upstream framing (sharper than the original draft)

Upstream already has `_maybe_override_dynamic_sd_cudagraph_mode()`, which
downgrades FULL graphs to PIECEWISE for exactly this reason — but its
`use_v2_model_runner` early-return makes it unreachable for dspark, which is
V2-only, and its own log line tells users to set that variable to get full
graphs. Upstream `main` still drafts `n_spec = num_speculative_steps` while
capture strides come from the schedule, so this is a reportable bug rather than
an unported fix. The upstream drafts are parked locally and are not posted.

## The ceiling is a separate, cheaper defect

A step whose token count exceeds `max_cudagraph_capture_size` silently falls back
to eager — same acceptance counters, far less throughput.

| conc | tokens (8 x conc) | vs ceiling 72 | agg tok/s |
|---|---|---|---|
| 8 | 64 | graphed | 205.90 |
| 9 | 72 | graphed (at the limit) | **215.87** |
| 10 | 80 | **over -> eager** | **167.39** |
| 12 | 96 | over -> eager | 163.93 |

`MAX_NUM_SEQS=10` caps concurrent requests at 10, so a busy engine is **always**
at the 10-request cap: at ceiling 72 the deployment was permanently parked on the
eager side of that cliff, and every number reported up to then had been measured
at conc=8 — inside the ceiling — which is why it stayed invisible across two
deploys. The correct ceiling is `MAX_NUM_SEQS x (1+k)` = 80, which is also the
maximum reachable decode batch; higher only spends capture memory on shapes that
can never dispatch.

Paired A/B, 2 reps per arm, alternating boot order to cancel the box's ~5%
per-10-boot drift:

| arm | correct | conc8 | conc9 | **conc10** | conc12 |
|---|---|---|---|---|---|
| ceiling 72 (control) | 30/30 | 204.70 | 213.30 | **164.70** | 167.56 |
| ceiling 80 (deployed) | 30/30 | 202.56 | 212.42 | **227.13** | 221.50 |

conc10 **+37.9%**, conc12 +32.2% (a 12-request burst queues at the 10-request
cap, so it is the same 80-token batch — the ladder never needs 96), conc8/conc9
unchanged within noise, acceptance windows identical, capture cost 0.57 -> 0.58
GiB/rank.

Three self-corrections worth keeping, because all three were confident and all
three were wrong: (a) "the remaining headroom is not the bottleneck" was inferred
from conc=8 when the operating point is conc=10; (b) a note claiming conc=12 needs
a ceiling of 96 ignored that `MAX_NUM_SEQS` means a 12-request burst never forms a
12-batch; (c) an earlier note claimed static `k=7` "wedges the GPU at concurrency
>= 6", which kept deployments parked at conc=8 — W53 ran `k=7` at concurrency 8
across four boots with no wedge, and the soak has since run it oversubscribed
(conc=16 then conc=8) for 3894 s, so that claim is withdrawn. All three were
caught by measuring, not by reasoning.

**Rules of thumb:** set `max_cudagraph_capture_size = MAX_NUM_SEQS x (1+k)`, and
always measure the cell at the `MAX_NUM_SEQS` cap. `setup.sh` enforces the first
rule and refuses to start if it is broken; `verify.sh` re-checks that the capture
ladder actually *contains* the value, since echoing the number is not proof.

### Two passes, two counts — do not quote one of them alone

The engine log shows **two** capture passes, so a claim about "how many shapes" is
meaningless without saying which:

| pass | log line | shapes |
|---|---|---|
| target model | `Capturing CUDA graphs (FULL)` | 10 |
| dspark drafter | `Capturing dspark CUDA graphs (FULL)` | 9 |

Both are built from the **13-size** configured ladder
`[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]` that the engine prints as
`cudagraph_capture_sizes`. Neither progress bar counts that ladder directly, so
quoting `9/9` as "the ladder has 9 sizes" is wrong, and so is quoting 13 as "13
graphs were captured". `verify.sh` asserts on the configured ladder because that
is the list the saturated batch size has to appear in. Total cost is 0.58-0.60
GiB per rank for both passes together, in ~6 s.

## Open

Soak status: the deployed config (static `k=7`, `FULL_DECODE_ONLY`, ceiling 80,
ladder `[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]`) completed **6 of 6
sustained load cycles with 0 wedges and 0 generation timeouts** (~3894 s of
continuous conc=16/conc=8 load, TP2, `restart_count` stayed 0). That is
like-for-like with the W27 mitigation baseline (0/6).

**What that does and does not prove.** The wedge is rare and load-shaped, so 6
clean cycles *lower* the risk rather than eliminating it; absence of a wedge in a
soak is not evidence of its absence in production. Keep
`CCL_SYCL_ALLREDUCE_TMP_BUF=1` (`scripts/serve.sh`) — it is the only mitigation
found (W27 AR 0/6 vs 4/5 same-run controls, Fisher p=0.0152), and the wedge does
not self-heal (53/53 restarts=0): a wedged engine keeps answering `/health` 200
while generating nothing, which is exactly why the soak gates on generated tokens
rather than on liveness.

Harness: `scripts/soak.sh --cycles 6 --cycle-seconds 600`. It treats **only
rc=124** as a stall and aborts inconclusive on an operator kill (rc=143), so a
PASS can only come from cycles that ran to their sustained duration. An earlier
version recorded its own kill as a timeout and reported a healthy engine as
broken; the reverse failure — a PASS tallied from a run the harness itself ended
— is the one actually worth guarding against.
