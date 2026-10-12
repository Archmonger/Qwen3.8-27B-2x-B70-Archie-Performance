# How to read the recorded metrics

The table in
[the README](../README.md#qwen38-27b-fp8-recorded-performance-metrics) is the
summary; this document is the reasoning that makes each row quotable, and the record
of the claims this investigation already got wrong so they are not re-derived as new
findings.

## Where each recording comes from

Every cell in the README's grid is the **median of three reps of a harness that is in
this repo**, taken in one session against one running container, on an engine that had
been attested first (`scripts/verify.sh`). One number per cell by design: no ranges, no
pooled means, and every value traceable to a script someone else can run.

That is a deliberate change from how the first version of this table was assembled, and
the reason for the change is worth keeping. The earlier table mixed three different
kinds of number in one column — an establishing paired A/B, later harness re-runs, and
within-boot reps — and while each was defensible alone, the column as a whole could not
be re-produced by any single command. Some cells came from scratch scripts that were
never committed. So when the harnesses changed, several published cells could not be
traced to shipped code at all. The rule now is:

- **A published number comes from a harness in `scripts/`,** not from an ad-hoc script.
  If the measurement needed a new script, the script ships.
- **Three reps per cell, median published.** Rep count and per-rep spread are recorded
  alongside, so a cell that quietly rests on one good rep is visible.
- **The engine is attested in the same session** as the measurement, not in an earlier
  one. Attestation reads what the engine printed about itself, so a container restarted
  or reconfigured between attestation and measurement invalidates the pairing.

Two cells are not token-generation rates and are produced differently:

- **Prompt Processing** is a *slope*, `d(tokens)/d(TTFT)` from
  `prefill.py --length-sweep`, not a tokens-per-second read off one request. See the
  next section for why the single-request ratio is not a prefill rate.
- **TTFT** is the median over the four test prompts from `prefill.py --test …`, each
  sample paired with its own immediately-preceding baseline reading.

The acceptance windows landing on the **same** draft and accepted counts across every
recording (`C1 26 169 7.500`, `NAT 256 512 3.000`) is the strongest signal that a
clean build boots the same engine: those counters are a fingerprint of the drafter,
the target, and k together.

Use `../scripts/bench/run-all.sh` to take the whole table again in one go.

## Why the prefill rate is a slope

The intuitive way to measure prefill throughput is one request: divide
`prompt_tokens` by TTFT. On this deployment that produces a number that cannot be true.
A 15-token prompt measured **182 ms** and a 347-token prompt measured **169 ms** —
which would mean prefiling 332 more tokens took negative time. It does not: at these
sizes TTFT is dominated by a per-request latency floor (queueing, scheduling, sampling
setup, SSE plumbing), and the actual prefill work is a minority of it. Dividing by a
number that barely contains prefill does not measure prefill.

The fix is to make the floor cancel. `prefill.py --length-sweep` sends prompts at
ten lengths from 16 to 8192 tokens and fits `tokens = slope x TTFT + c` by least
squares; the slope is the marginal cost of tokens, and the intercept is the floor
(about **91-96 ms** here, published as the fitted constant rather than inferred). The
fit is restricted to samples with `TTFT >= 250 ms`, because the short end is
floor-dominated and would drag the slope toward the intercept.

Two contamination traps found while building it, both of which produced confident
wrong numbers first:

1. **Cross-run prefix-cache hits.** The first two sweeps returned **2917.7** and
   **6664.2 tok/s** — a 2.3x disagreement, all of it in the probe. The prompt salt was
   a per-process counter, so the second run rebuilt byte-identical prompts and got
   cache hits on the longest ones, deleting the prefill it was supposed to be timing.
   Salts are now unique across processes (pid + wall clock), and a cache hit is
   *detected* via `cached_tokens`, reported, and excluded — a cache-hit sample is not a
   slower measurement of the same thing, it is a measurement of a different thing.
   After the fix the three reps gave 2840.1 / 2818.7 / 2808.3, a 1.1% spread, with
   zero excluded hits.
2. **Session drift read as a prompt property.** Measuring all samples of one test and
   then the next made TTFT fall monotonically in *run order* (225 / 207 / 168 / 166 ms)
   with ~2 ms spread within each test — the ordering was the variable, not the prompt.
   Samples are now interleaved round-robin with the order alternating per round, and a
   baseline prompt is measured immediately before every sample.

That second fix surfaced a guard that vetoes valid measurements, which is its own kind
of bug. The stability gate originally tested the *baseline's* spread, and flagged
planning and prose UNSTABLE in every rep because their baseline alternated between
~186 ms and ~141 ms — while the actual published readings moved only ~3 ms
(207.4-211.0 and 207.0-210.3) and did not track the baseline's state at all. The
baseline's mode turned out to depend on which prompt had run immediately *before* it,
a property of the probe sequence rather than of anything under test. The gate now tests
the spread of the quantity actually published, and the baseline is reported as
diagnostic context (`baseline_spread_ms`, `baseline_ms_samples`) instead of wielding a
veto. Same lesson as `--max-tokens` below: gate on what you publish.

## Prefix caching is engaged but inert at these sizes

The engine reports `enable_prefix_caching="True"`, so it is tempting to credit cached
prefill for anything fast. It is not doing that here, and the evidence is
`prompt_tokens_details.cached_tokens` on repeated identical requests:

| prompt tokens | `cached_tokens` on a repeat |
|---|---|
| 296 (planning/prose) | 0 |
| 347 (coding) | 0 |
| 1292 | 0 |
| 2572 | 0 |
| 6412 | 4096 |

Reuse begins only once a prompt fills at least **two full 2048-token KV blocks** —
`max(0, (n // 2048 - 1) x 2048)` fits every row above. Worth knowing because the
engine's own config line is misleading in two ways: `block_size` is *requested* as 64
and *reported* as `2048` in `cache_config_info`, and the log carries "Prefix caching in
Mamba cache 'align' mode is enabled ... experimental" alongside a
"Prefix cache hit rate: 0.0%" at every interval.

Do not lean on the engine's aggregate **"Prefix cache hit rate"** log line instead. It
is cumulative over the container's life and it is *not* 0.0% on a box that has been
probed: after a session of sweeps it read up to 12.3% (1071 of 1700 log intervals
nonzero), entirely from long-prompt probe traffic — including the contaminated sweep
runs described above, which were excluded from the fit but still registered hits. The
per-request `cached_tokens` field is the only figure that attributes reuse to a specific
measurement; the aggregate line says only that something somewhere hit.

Two consequences for anyone measuring here. Any causal claim that rests on "prefix
caching makes prefill cheap" is unsupported at prompt sizes below ~4k tokens — which
includes every prompt in this table. And any probe that reuses prompt text across runs
must check `cached_tokens`, because the contamination only bites at the long end, which
is exactly where a slope is anchored.

## `--max-tokens` is part of the metric, not a knob

`conc.py`'s `aggregate_tok_s` divides counted `completion_tokens` by wall time, but
wall time also pays for prefill. The test prompts are a fixed few hundred tokens
(server-reported `prompt_tokens`: 286 counting, 296 planning, 296 prose, 347 coding),
so that uncounted prefill cost is amortised over `max_tokens`: at `conc=10`, 96
tokens/request reads ~13% below 256 on the *same boot*, while `conc=1` moves only
~4%. That asymmetry is what made it look like a host problem.

Two things that used to be written in this section are **wrong and must not be
re-derived**. First, the prompt was quoted as "~243 tokens", which is the `wc -w`
*word* count of the planning prompt file, not a token count; the server's own figure
is 296. Second, the single-stream immunity was attributed to "prefix caching making
prefill nearly free", and prefix caching does not engage at all at these sizes (see
[Prefix caching is engaged but inert at these sizes](#prefix-caching-is-engaged-but-inert-at-these-sizes)),
so it cannot be making anything free in either cell. The size of the effect
is measured and is the whole justification for the guard; *why* single stream is
nearly immune while the capped cells move is not established. Quote the ~15%, not a
mechanism.

A verification pass measured the batched cells at **197** against the establishing
A/B's **227.13** and called the gap an unidentified ~13% host-side depression. That
diagnosis was wrong, and the numbers in this section are what caused it: the pass
invoked `conc.py` directly, which fell through to a `--max-tokens` **default of 96**,
while the establishing A/B and both scripted callers (`run-all.sh`, `soak.sh`) used
**256**. Re-running one boot with alternating arm order — the only change being the
flag:

| `--max-tokens` | 96 | 256 | establishing A/B |
|---|---|---|---|
| conc=10 | 193.23-202.94 | **229.16-231.64** | 227.13 |
| conc=1 | 42.37 | 44.32 | 42.6 |

The same boot reproduces the establishing A/B at 256. Nothing was wrong with the
engine, the image, or the host: 0 preemptions, 0 KV usage, 0 restarts, correct graph
capture throughout. `conc.py` now defaults to 256, prints `max_tokens` with every
cell, and refuses a non-canonical value unless you pass `--confirm-noncanonical`, so
this comparison can fail loudly instead of quietly measuring a different metric. The
natural probe had the same failure mode (a 50.21 reading from a `--max-tokens 512` run
compared against a ~59 taken at 96), which is why its row is quoted at the canonical 96.

**A metric that moves ~15% on an unrecorded parameter is not a metric.** Quote the
harness, the cells, *and* `max_tokens` with any aggregate number. The mechanism, the
guards, and how to prove the guard can fail are recorded as rule 3 of the harness
bugs in [verification.md](verification.md#harness-bugs-that-misreported-a-healthy-engine).

Dead end, kept so nobody re-walks it: an orphaned root-owned `intel_gpu_top` holding
the GPU perf/OA unit open was blamed for the depression, then cleared — it started
Oct 3, so it was also running during the Oct 7 run that recorded 227.13 and cannot
explain a delta between the two. The real cause was in the harness all along.

## The two measurements that changed the number

1. **`MAX_NUM_SEQS x (1+k)` is the graph ceiling, and 72 was wrong.** A decode step
   whose token count exceeds `max_cudagraph_capture_size` silently falls back to
   eager. With `MAX_NUM_SEQS=10`, the saturated batch is `10 x 8 = 80` tokens, so at
   ceiling 72 the only state a busy engine ever reaches was ungraphed. Paired A/B
   (2 reps/arm, alternating boot order): conc10 **164.70 -> 227.13 (+37.9%)**,
   conc12 167.56 -> 221.50 (+32.2%), conc8/conc9 unchanged. Details:
   [cudagraph-stride-defect.md](cudagraph-stride-defect.md).
2. **Graphs + DSpark is safe only at a consistent stride.** Static `k=7` with
   `FULL_DECODE_ONLY` is 30/30 correct; the dynamic schedule that looked faster
   produced 6-13/30 correct while acceptance counters stayed healthy. Same document.

## Spread, and what is not comparable

Run-to-run spread is real and metric-dependent. With every harness at its canonical
parameter, repeated reps **on one boot** gave the C1 probe a ~0.6% range and the
natural probe (canonical `--max-tokens 96`) a ~0.4% range — see
[recorded-metrics.md](recorded-metrics.md) for the per-rep values.

That is reps-on-one-boot, which is *not* a session-to-session noise floor and must
not be quoted as one: it holds the boot, the box state and the cache state fixed, and
this host drifts ~5% per 10 boots. Measure the floor for the
comparison you are actually making, by booting the identical config as a calibration
arm. Anything under ~3 tok/s on the natural metric needs repeats, and real A/Bs are
paired with alternating arm order rather than run once each.

Two figures from the investigation were at one point passed off as spread and were not
spread at all. The ~13% batched gap is a `max_tokens` mismatch (above). The
**47.70-55.44 C1 / 21.37-24.85 natural** range is a genuine measurement, but from the
**eager era** — k=4, graphs off — so it is not spread around the graphed numbers at
all; it is the era-mixing the next section warns about, and the numbers are kept here
only as a labelled dead end. Before attributing a gap to noise, check
that both sides used the same `max_tokens`, the same cells, the same probe, and the
same era.

## Quote the era and the metric name with every number

The realistic-suite metric for this configuration is **~42 tok/s**
(`tok_s_1_100_after_ttft`, per-run medians 42.03-42.55). Do not compare
that against the 15.67 that appears in the investigation's working notes:
15.67 was measured in the **eager, k=4** era, before graphs were enabled at all, and
it is not comparable to anything in the README table. Likewise the "~1.6x from spec
decode" figure (13.76 DSpark k=4 vs 8.60 no-spec on `vllm bench serve`) and the
spec-off suite baseline of 10.32 are eager-era, k=4 measurements — useful as a
same-era A/B, useless as context for the graphed config.

Always state which metric you are quoting. `tok_s_1_100_after_ttft` (`suite.py`),
`aggregate_tok_s` (`conc.py`), `median_tok_s_after_ttft` (`probe.py`), and the
engine log's "Avg generation throughput" are four different measurements.

## Prove the knob engaged before reporting "no effect"

An A/B where the knob did nothing and an A/B where the knob did nothing *useful*
produce the same table. Both read "no effect", and only one of them is true. The
earliest version of this project reported three `XPU_*` environment knobs inert on
throughput alone; the graph knob's inertness was later demonstrated properly (the
engine log showed `cudagraph_mode=NONE` and no capture, so the flag had genuinely
not applied), but the other two rested on "the number did not move". A flat result
is only evidence about a knob you showed was live.

So `scripts/bench/arm.sh` requires a positive engagement assertion — `NEEDLE` /
`NEEDLE_ABSENT` — and aborts the arm rather than recording a row when the engine
log and the dispatch census both fail to match. Two rules make that assertion
worth anything:

**Read the engine's own words, not your intent.** Attestation already does this
for configuration (`../scripts/lib/attest.sh`), and the same discipline applies
to code paths: grep the log for what the kernel printed, and census the dispatch
branches actually taken.

**Place an instrument at the point of no return.** This is the trap that nearly
produced a published falsehood. A census note added at the *top* of the ESIMD GDN
entry points logged every time the function was entered — before the block that
checks `conv_state.dtype` and `ssm_state.dtype`. It printed confidently from an
arm where the guard then rejected the call and the kernel never ran. Moving the
same note immediately before the `op()` invocation inverted the result: the
combination of `--dtype float16` and `--mamba-ssm-cache-dtype float16` prints it,
and `--dtype float16` alone does not. A probe that fires on entry proves the
function was *called*; only a probe beside the launch proves it *ran*.

The corollary is that the census needs its own mount proof. The instrumented
modules are bind-mounted over the image's copies, and a failed mount still
produces valid `tok/s` — it just produces no census lines, which is exactly the
evidence used to declare a path dead. `arm.sh` therefore asserts the live files
contain the probe marker before it believes their silence.

**Engagement proves a path ran. Only ablation proves it mattered.** Those are
different claims and the second does not follow from the first. A shipped comment
asserted that `--dtype float16` is faster *because* two ESIMD kernels are gated on
`float16` and never run under bfloat16. Both halves checked out — the census showed
the kernels firing under fp16 and silent under bf16, and the engine was faster under
fp16 — and the conclusion was still wrong. The image exposes
`VLLM_XPU_DISABLE_ESIMD_NORM` and `DISABLE_ESIMD_GDN_SPEC`, so each kernel could be
switched off with the dtype pins still set, and the census proved each ablation
applied. The win did not move. Disabling both at once did not move it either. The
kernels were correlates.

Two things made that mistake easy, and both are guardable:

*The explanation was available before it was tested.* The source contains a literal
`x.dtype == torch.float16` in the kernel's gate, so the causal sentence read like a
description of the code rather than a claim about performance. A gate in the source
establishes that a path is *reachable only* under fp16; it says nothing about how
much of the time that path is responsible for.

*Ablate jointly, not only singly.* Each kernel off alone changed nothing, which is
weak evidence on its own: two redundant paths can each be individually unnecessary.
The arm that settles it disables both. If a single ablation shows no effect and the
paths might be redundant, the combined arm is not optional.

The rule: when a knob's effect is explained by some path that newly appears, look
for the switch that removes *just that path* and measure again. If the image exposes
no such switch, write that the mechanism is unresolved instead of publishing the
correlation as the reason.

### A harness fixture must have the shape of the real artifact

Engagement checks read files, so the harnesses feed them fixtures. If a fixture is
an *easier* version of the real file, the harness goes green against code that cannot
parse reality. That happened while separating stdout from stderr in `bench/arm.sh`:
the pipeline that flattened probe output was replaced with a single-line brace match,
and every arm would have been discarded as unparseable, because `probe.py` prints
**indented** JSON and a one-line `{.*}` match returns the lone character `{`. The
falsification suite stayed green throughout, because every fixture in it was written
as one-line JSON.

Three consequences worth carrying forward:

* When a parser is added or changed, at least one fixture must be the real artifact's
  actual shape -- indented, surrounded by log lines, whatever the producer truly
  emits. `falsify/arm-cases.sh` now pins the parser against probe.py's multi-line
  form directly, not just against compact fixtures.
* A parser must fail *loudly*. An import error or empty result that exits nonzero with
  no message is indistinguishable from "this measurement is degenerate", so a plumbing
  fault rejects a healthy boot and nobody can see why.
* In a mutation harness, an environment variable must be `export`ed on its own
  statement. `VAR=value . file` does not export `VAR` -- bash treats an assignment
  preceding the source builtin as a plain shell variable and rolls it back when the
  source returns. A mutant that dies on a missing variable reports "still rejected",
  which the harness then reads as "this rule is not load-bearing": a verdict about the
  mutation caused entirely by the harness. The same failure mode as an unwired
  assertion, but it looks like evidence rather than silence.

## The INT4 placeholder cells

The INT4 table in the README is empty on purpose. Until those cells are filled from a
run this repo produced, do not substitute numbers from elsewhere: the INT4 figures
recorded during the investigation (C1 71.08, natural 21.19 with DSpark `k=7`, no-spec
10.93 / 11.22, KV 19.7 GiB, 35/35 correct) were measured on the base image with a
**dynamic** schedule and **no graphs**, which is not a configuration this repo can
currently stand up — the dynamic schedule is exactly the shape that garbles output
under graphs, and a no-graphs run is a different era, so those figures fail the
era-matching rule above as well. The constraints that decide which INT4 checkpoint and
which `SPEC_METHOD` are even legal are in
[`../profiles/int4/README.md`](../profiles/int4/README.md).

## Third-party numbers on this hardware that are not parity targets

These are other people's published figures, recorded here only so nobody re-derives
them as a regression:

- The **86.93 / 94.58** suite records come from a *different checkpoint*
  (`Qwen3.8-27B-Uncensored-int4-AutoRound`) on a *patched vLLM fork* with
  `VLLM_XPU_LM_HEAD_INT8` / `VLLM_XPU_DRAFT_LM_HEAD_INT4` /
  `VLLM_XPU_HOST_STAGED_COLLECTIVES` — env vars that exist in neither stock vLLM
  nor the llm-scaler image. The same checkpoint scores 48.74 without that patch
  stack, so the patch stack carries most of the number.
- The **72.2 tok/s** C1 figure attributed to PR #620 is, on this box, 58.75-59.74
  on the author's own image (production reaches 54.94 at that shape). The claimed
  72.2 is attributed in his own commit message to a *fine-tuned drafter*, a
  different artifact from the released checkpoint measured here. Separately, that
  PR's real delta is 3 commits / 13 lines, and **0.26 already contains all three on
  the dspark path**, so no image patch is needed — which is why this repo ships none.
