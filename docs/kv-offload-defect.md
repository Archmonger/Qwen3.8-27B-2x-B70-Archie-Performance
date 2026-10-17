# The KV-offload connector defect

Status: **root-caused to the connector; signature 1 fixed by a ported upstream patch.**
**The repository ships it off, everywhere, and that is the whole rule.** Both
`scripts/serve.sh` and `profiles/fp8/fp8.env` set `KV_OFFLOAD_SIZE=0`, which omits
`--kv-offloading-size` entirely. Nothing a reader clones from this repo inherits a
tier that has never served a single read here and once took the engine down on one
long request.

One box runs it at 50 GB, and the way it does so is the part worth copying if you
want the same: a gitignored `profiles/fp8/host.env` containing `KV_OFFLOAD_SIZE=50`.
It is read after the profile by every script that loads one and handed to compose as
the final `--env-file`, so the value attested and the value baked into the container
cannot disagree; it may set that one key and nothing else. The reason it is not an
uncommitted edit to `fp8.env` is that such an edit is silently reverted by the next
`git checkout` while every check still passes — a config that changes underneath you.
Full mechanics in [configuration.md](configuration.md).

Enabling it at all is only safe behind the hotfix in
`image/patches/kv-offload-52923.patch`, which this build needs because its vLLM
commit predates the upstream fix; `setup.sh` refuses to start the tier-on/unpatched
pairing rather than leaving it to fail on a long request.

Do not treat the tier as a throughput knob: on this workload it serves no reads (see
below), so treat it as capacity headroom. And do not enable it on an image that has not
been through the patched Dockerfile.

Host these numbers came from: 2x Intel Arc Pro B70, TP2, XPU, image
`vllm-llmscaler:icr26.35`, vLLM `0.26.1.dev0+g568afb3a1.d20260907`, target
`Qwen3.8-27B-FP8`, drafter `qwen3.8-27b-dflash-drafter-fp8-b70`, static `k=7` with
`FULL_DECODE_ONLY` graphs. 188 GiB host RAM.

## The symptom

The engine dies — not slows, not degrades, dies — while serving **long-context**
requests. Short and medium traffic is entirely unaffected, which is why the profile
passed `scripts/verify.sh`, a correctness battery, and hours of ordinary traffic
before this showed up. Nothing in the pre-deploy gate exercises a 100k+ prompt.

Two distinct signatures were observed, both from a container started with
`--kv-offloading-size 50`:

| # | where it dies | what the kernel says |
|---|---|---|
| 1 | `EngineCore` raises `AssertionError` in `kv_connector/v1/offloading/scheduler.py:975` — `assert len(offload_keys) == len(offload_block_ids)` — reached from `scheduler.schedule()` → `_build_kv_connector_meta` → `build_connector_meta` → `_build_store_jobs` | nothing; this is a Python logic fault, the GPU is not involved |
| 2 | `Worker_TP1` raises `RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` from `torch/xpu/graphs.py:107 replay()` under `dflash/speculator.py:443 propose()`, and from `copy_event.synchronize()` in `v1/worker/gpu/async_utils.py:50` | `xe 0000:83:00.0: [drm] Tile0: GT0: Check job timeout … Timedout job … trying reset from guc_exec_queue_timedout_job`, then `reset done`. **A GPU engine reset, visible in the kernel log, at the same second as the vLLM error.** |

A third signature appeared once, earlier in the same investigation, from the same
config: `error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)` from
`async_copy_to_np` (`x.to("cpu", non_blocking=True)`) and from
`kv_block_zeroer` (`idx.copy_(ids_pinned[:n], non_blocking=True)`). Both are
device→host staging copies, i.e. host-side allocation, and the KV pool was **6.5%
full** (`kv_cache_usage=0.06457`) when it fired, so nothing was being denied KV
blocks. That reading is what sent the investigation down the VRAM path first; see
"What was falsified".

The container's `restart=unless-stopped` policy means signature 1 and 2 look
different from outside: after 1 the container exits and restarts, after 2 the
APIServer survives while both workers are dead, so **`/health` keeps returning 200
against a zombie engine**. Probe with a real completion, never with `/health` alone —
this is the `health-200-from-the-wrong-engine` trap in another costume.

## The A/B

Same host, same image, same argv, same harness; the only difference is whether
`--kv-offloading-size` is present at container-create time (see
[configuration.md](configuration.md) — the value is baked into `Config.Env` at
create, so this requires a **recreate**, not a restart).

Prompt sizes below are `prompt_tokens` as the server reported them, not the driver's
nominal word count.

| | flag present (`50`) | flag omitted (`0`) |
|---|---|---|
| single ~160k-token prompt | died on all 3 attempts: `AssertionError` at 06:34:39, `DEVICE_LOST` at 06:39:08 (the clean case, one request alone in flight), `DEVICE_LOST` at 07:41:22 on the re-armed arm | passed 3/3, `restarts=0` |
| two concurrent ~160k-token prompts | not measured on this arm — the single-request cell killed the engine first, and it was not re-run once the engine came back | passed 4/4 (2 reps x 2) |
| 244,510 prompt tokens (93% of the 262k window) | not reached | passed, `ttft` 230 s |
| host `MemAvailable` | 8.2–8.8 GiB | 80–88 GiB |
| `restarts` after the run | 1–2 | 0 |

The 06:34:39 attempt had two requests in flight (`Running: 2 reqs` in the engine log,
from a duplicate probe driver left running), so it is not a clean single-request
reproduction; 06:39:08 is. The field failure that prompted this investigation was also
a single request: the last engine sample before it read `Running: 0 reqs`, one
completion returned 200, and the next line in the log is the error, ~2 h after the
engine had gone idle.

Both crash signatures are therefore downstream of the same switch, and omitting the
flag removes both.

Separately, to confirm the surviving config does the work rather than merely not
crashing: a needle-in-a-haystack probe on the flag-omitted boot placed a planted
constant mid-way through a 160,279-token prompt and the model returned it exactly
(`4211`). This was only ever run on the working arm — there was no engine left to
test on the other one — so it establishes that long context is *correct*, not that
it is a differential.

Both long-context results above came from one boot each. The flag-omitted config also
passes the repo's own gates, which is what says the fix did not silently break the
measured configuration: `scripts/verify.sh` → correctness 15/15, acceptance
`C1 26 169 7.500` / `NAT 272 512 2.882`, `VERIFY: PASS`, and
`scripts/falsify/run-all.sh` → every harness green. Each harness attests what it saw
on its own boot; the numbers above are from the boot that served the long-context
ladder. (No count is quoted: this line said "8/8" when the suite held 7.)

## Why it is safe to leave off: the tier never served a read

The connector is a *cache*, and a cache whose reads never hit is pure cost. Across
every boot measured with the flag present:

- `vllm:kv_offload_cpu_cache_read_usage_perc` — `0.0`, every sample, no exception.
- `External prefix cache hit rate` — `0.0%`, every sample.
- `kv_offload_cpu_cache_write_usage_perc` — climbs monotonically (`0.0 → 0.0049 →
  0.0098 → 0.0146`). It writes and writes and never reads.

It is not free, either. It is not the GPU that pays — the KV pool is **identical**
with and without the tier. The A/B ran at `GPU_MEM_UTIL=0.91` and measured
`10.29 GiB / 403,165 tokens / 1.54x` in both arms; the shipped `0.92` boot with the
tier off reproduces `10.59 GiB / 415,029 tokens / 1.58x`, the figures
[configuration.md](configuration.md) publishes for the shipped `float16` default. So
this is not a memory-for-memory trade.

That identity is not inferred from the A/B alone: it was re-confirmed on the one box
that runs the tier. Its live `profiles/fp8` container boots with `kv_offloading_size:
50.0` in its non-default args — from `host.env`, not from the committed profile — and
reports `Available KV cache memory: 10.59 GiB` / `GPU KV cache size: 415,029 tokens`,
the same pair, with the tier **on**. So the KV capacity figures in
[configuration.md](configuration.md) hold with the tier on, because the pool they are
read from did not change.

What that does **not** license is a claim about the README's throughput table. Those
numbers were recorded with the flag omitted, and this file has no tok/s A/B across
the switch. The strongest tier-on evidence held is behavioural rather than a rate:
`scripts/verify.sh` against the live tier-on container reproduces the tier-off
acceptance fingerprint exactly — `C1 26 169 7.500` / `NAT 272 512 2.882`, the same
draft counts and mean accepted length, with correctness 15/15 and serving surface
5/5. Identical acceptance geometry and an identical KV pool make a throughput
regression unlikely, and that is why the table was not re-measured; it is still an
argument, not a measurement, and the table's own provenance note already says the
absolute tok/s figures are one box's. Re-measure with `scripts/bench/arm.sh` before
quoting them as tier-on:

```bash
# Production owns both GPUs, so stop it first; two engines on one pair both
# mis-size the KV pool and the numbers mean nothing. LABEL and PORT are
# positional, and PORT must not be production's 8103.
scripts/bench/arm.sh tier-on  8199
scripts/bench/arm.sh tier-off 8199 KV_OFFLOAD_SIZE=0
```

`arm.sh` seeds the tier from the profile, passes it to the container, prints it on
the launch line as `tier=<n>GB`, and `attest.sh` refuses an arm whose engine
disagrees — all three of which were missing when the tier was switched on, and each
of which on its own is enough to produce a table of tier-off numbers labelled as
the shipping config.

The **host** pays, and it pays progressively: the store is not a boot-time
reservation but grows with offload
activity. One boot measured 87.6 GiB available at ~2 minutes old, still loading
shards, and 8.8 GiB after the verify pass had driven traffic through it. A fresh boot
with the flag omitted sits at 80–88 GiB after the same work. The store does not
appear in any process RSS (worker `VmRSS` 4.82 GiB, `VmHWM` 6.52 GiB) or in the
container's cgroup (`memory.current` 11.5–12.1 GiB, `anon` 11.5 GiB) — it is held
outside what either of those accounting views shows, which is why the shortfall only
becomes visible as `MemAvailable` collapsing.

So the honest description of the trade is: **it consumes host RAM proportional to
use, serves no reads on this workload, and kills the engine at long context.**

`offload_prompt_only` defaults to `True` (`v1/kv_offload/base.py:510`), which is why
prompt blocks are the bulk of what it stores — and why the cost scales with prompt
length, which is exactly the axis this defect breaks on.

## Upstream

**Correction, recorded because an earlier version of this file got it wrong.** This
section originally said "unfixed at the time of writing" and pointed at #55503, #56795
and PR #53648. Those are real related reports, but they are not the fix, and the fix
did exist: **[vllm-project/vllm#52923](https://github.com/vllm-project/vllm/pull/52923)**
"[Bugfix] Wait for offload keys before storing chunks" was merged 2026-09-03 (merge
commit `092334c82`, fixes #52170). The mistake was searching for the assert text among
open issues instead of the merged history. Re-checking upstream is cheap and this is
the cost of not doing it.

The reason the defect is live here despite that merge: the image pins
`intel/llm-scaler-vllm:0.26.0-b2`, whose vLLM commit `568afb3a1` is dated 2026-07-26 —
five weeks before the merge. The `.d20260907` in `0.26.1.dev0+g568afb3a1.d20260907` is
the *build* date, not the commit date, which is what made it look current. Check the
commit, not the tag.

The shipped hotfix is `image/patches/kv-offload-52923.patch`, applied in
`image/Dockerfile`. It is a **port, not the upstream diff**: this build's
`storable_chunks()` predates the PR's *base* as well, carrying neither of the two extra
bounds upstream applies, so the upstream hunk has no context to match against. The
ported function is byte-identical to upstream's post-fix version.

The build pins two hashes — the pre-patch file hash and the post-patch file hash — so a
moved base fails the build loudly instead of producing an image with a half-applied or
mis-fuzzed patch. `setup.sh` then attests the *running* image, because `setup.sh`
reuses an existing tag without rebuilding: edit the Dockerfile, run `setup.sh`, and you
get the old unpatched image. Enabling the tier against an unpatched image is fatal
there, and a warning while the tier is off (without the connector built the assert is
unreachable, so the tier-off case is not worth blocking).

The probe and its severity table live in `scripts/lib/kv-patch.sh`, and
`scripts/falsify/kv-patch-cases.sh` exercises that same code: classifier cases
(verdict under a torch warning on the same stdout stream, verdicts buried in prose,
empty output, both verdicts at once), the severity table over
state x `KV_OFFLOAD_SIZE`, and — when docker and the pinned base are present — the
shipped image (must read `PATCHED`), the unpatched base (must read `UNPATCHED`), a
mutant image with one bound `sed`-ed out (must read `UNPATCHED`), and a nonexistent
image (must read `UNKNOWN`). The mutant case is what shows the gate is not vacuously
green; without it, a probe that can only ever print `PATCHED` would leave the
refusal decorative. Case counts are not quoted, for the same reason they are not
quoted for `run-all.sh`.

What #52923 does **not** cover: it fixes the key/block desync, i.e. signature 1. See
"What is still not established" for signature 2, which the A/B tied to the same switch
but whose kernel-level mechanism was never isolated.

## Does the hotfix actually work here?

Yes, on signature 1 — measured, not inferred. `scripts/bench/kv-ladder.py` is the
gate: it replays the exact cells from "The A/B" above against the running engine with
`KV_OFFLOAD_SIZE=50` on the patched image.

| cell | pre-fix (same host, same argv; times are those quoted in "The A/B") | post-fix re-run, `kv-ladder.py`, 2026-10-16 07:40Z–07:54Z |
|---|---|---|
| single ~160k prompt x3 | **died 3/3** (`AssertionError` at 06:34:39, `DEVICE_LOST` at 06:39:08 and 07:41:22 — the earlier A/B boot, not this window) | `http=200`, `prompt_tokens=158803` x3, 113 s each, planted needle returned correctly on all 3 |
| two concurrent ~160k | not measurable — the single-request cell killed the engine first | both `http=200`, `prompt_tokens=158805`, 116 s / 226 s, needles correct |
| 93% of the window (target 243,793) | not reached | `http=200`, `prompt_tokens=241941`, 227 s, needle correct |
| `restarts` / crash lines after the run | 1–2 | **0 / 0**, no `AssertionError`, `DEVICE_LOST`, or `Traceback` in the window |

`cells=6 failed=none`, `restart_delta=0`, `err_delta=0`, at 19 GiB host
`MemAvailable`. Since the engine booted, the connector has written `2.4955e+11` bytes
GPU→CPU and read back `0.0` — it is doing the work, and still serving no reads,
exactly as the "never served a read" section predicts. (That counter is cumulative
since the 01:44Z boot, not the total for this one run.) That read/write pair is why
this tier is shipped as capacity headroom and not as a speed knob.

Two honesty notes about this table. First, the pre-fix column and this one are not
a same-boot A/B: the pre-fix figures are the earlier 50 GB boot quoted above, and
this re-run is one boot of the patched image. Second, every cell here is calibrated
against the server's **own** `prompt_tokens`, which is what caught two harness bugs
during this re-run — a hand-estimated chars/token asked for 160k and got 70,901, and
an unsalted prompt would have been served from the GPU prefix cache and skipped the
`_build_store_jobs` path entirely. Both would have "passed" while testing nothing.
That is why `kv-ladder.py` records `in_range` per cell and fails a cell whose token
count is off, and why every rep carries a unique prefix salt.

Signature 2 (`DEVICE_LOST` + xe GUC job timeout) is **not** claimed fixed: the fix is
a Python scheduler bound, and 2 of the 3 pre-fix deaths were signature 2, so 6 clean
cells are real evidence about signature 1 and weaker evidence about signature 2. See
"What is still not established".

## What was falsified (recorded so it is not re-litigated)

- **"It is VRAM exhaustion."** The error-39 sample had a 6.5%-full KV pool, and the
  KV pool is byte-identical with the tier on and off. Lowering `GPU_MEM_UTIL`
  0.92 → 0.91 as a mitigation cost ~12k KV tokens (`415,029 → 403,165`,
  `1.58x → 1.54x`) and did not prevent a single subsequent crash. It was aimed at the
  wrong resource.
- **"ECC is reserving VRAM."** Both cards are already ECC-disabled — `xpu-smi config
  -d N` reports `Current: disabled / Pending: disabled`, `dump --metrics
  ecc.mode.current` reports `Disabled`, and `torch.xpu` reports 32656 MiB, the
  ECC-off figure (ECC-on measures ~27904 MiB; Battlemage reserves 1/8). There was
  nothing to reclaim. Note `xpu-smi`'s summary-table ECC column reads the
  *capability* bit, not the state — trust `config`/`dump`.
- **"The `sync_prev_onload()` in `run_fullgraph()` is the racing partner."** A
  tempting read of `cudagraph_utils.py:400`, since the line immediately precedes the
  faulting `replay()`. It is a different subsystem: `get_offloader()` is imported from
  `vllm.model_executor.offloader.base`, i.e. **weight** offloading, and on this boot it
  is a `NoopOffloader`. It has nothing to do with the KV connector.
- **"oneCCL mitigation `CCL_SYCL_ALLREDUCE_TMP_BUF=1` is missing from the workers."**
  It is set (`serve.sh:69`); it is simply not visible in `/proc/<pid>/environ` because
  the workers call `setproctitle`, which rewrites that region. `comm`/`cmdline`
  reading `VLLM::Worker_TP0` is the tell. Read the parent's environ instead.
- **"Host RAM pressure is the root cause."** Host depletion is real and is the
  plausible mechanism for signature 2, but it is downstream of the same switch —
  it does not survive the flag being omitted — so it is not an independent cause.

## What is still not established

The micro-mechanism of **signature 2**. The A/B proves the connector is the operative
variable, but not *why* a GUC job on card1 stalls long enough to trigger the kernel's
job timeout and GT reset. The leading hypothesis is host page-in stall: with ~28 GiB
swapped and single-digit-GiB `MemAvailable`, a device↔host copy whose host pages are
no longer resident has to wait on the swap device before the GPU-side transfer can
complete. That fits the kernel evidence and the failure sites (all device↔host copies,
`x.to("cpu", non_blocking=True)` and a pinned-index `copy_`), but it was never
isolated, and it should not be quoted as a finding. No zero-copy host-transfer flag is
enabled on this deployment, so do not reach for one as a mitigation.

Signature 1 needs no such caveat — it is a plain invariant violation in the
scheduler, reproducible from the assert alone.
