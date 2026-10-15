# The KV-offload connector defect

Status: **root-caused to the connector, disabled by default.** `profiles/fp8/` ships
`KV_OFFLOAD_SIZE=0`, which omits `--kv-offloading-size` entirely. Do not re-enable it
without reading this file, and do not treat it as a tuning knob — on this deployment
it is a crash vector that buys nothing.

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
`scripts/falsify/run-all.sh` → 8/8 harnesses. Each harness attests what it saw on its
own boot; the numbers above are from the boot that served the long-context ladder.

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
tier off reproduces `10.59 GiB / 415,029 tokens / 1.58x`, the same figures the README
publishes. So this is not a memory-for-memory trade. The **host** pays, and it
pays progressively: the store is not a boot-time reservation but grows with offload
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

Unfixed at the time of writing. The assert in signature 1 is reported verbatim in
[vllm-project/vllm#55503](https://github.com/vllm-project/vllm/issues/55503)
(open, spec-decode + offloading, same call path); the sibling
[#56795](https://github.com/vllm-project/vllm/issues/56795) (open) describes
`next_stored_block_idx` advancing for every KV group when no keys were stored, which
is the same key/block desync from the other side. The candidate fix
[#53648](https://github.com/vllm-project/vllm/pull/53648) ("Fix offloading scheduler
chunk bounds race") is **not merged**. Worth re-checking those three before deciding
the tier is permanently unusable — but nothing here should wait on them.

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
