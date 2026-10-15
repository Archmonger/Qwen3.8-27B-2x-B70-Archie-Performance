# Qwen3.8-27B-FP8 configuration

Every value below is what the run that produced
[the recorded metrics](../README.md#qwen38-27b-fp8-recorded-performance-metrics)
actually used, and what `setup.sh` + `scripts/serve.sh` re-create. `[HW]` marks an
item someone on a slightly different box may plausibly have to change.

`scripts/verify.sh` checks these against **the engine's own printed config** rather
than your env file, because an env var vLLM silently ignored looks identical to one
that took effect.

| if you want | read |
|---|---|
| the image recipe and why it is rebuilt | [driver-overlay.md](driver-overlay.md) |
| the drafter pairing and its config patch | [dspark-drafter.md](dspark-drafter.md) |
| the `dri-by-path` directory | [dri-by-path.md](dri-by-path.md) |
| the oneCCL wedge, P2P, and the inert knobs | [oneccl-and-p2p.md](oneccl-and-p2p.md) |
| the graph-stride defect and the ceiling rule | [cudagraph-stride-defect.md](cudagraph-stride-defect.md) |

## Pinned image

| item | value |
|---|---|
| local image | `vllm-llmscaler:icr26.35` (18.3 GB, built by `setup.sh`) |
| base | `intel/llm-scaler-vllm:0.26.0-b2` @ `sha256:52218ad8…f641fe3b` |
| driver source | `intel/omix:0.4.0-devel-ubuntu24.04` @ `sha256:ee48e296…970ef0b3` |
| engine | vLLM `0.26.1.dev0+g568afb3a1.d20260907`, Python 3.12 venv at `/opt/venv` |
| kernels | `vllm-xpu-kernels` + `custom-esimd-kernels-vllm` from the base, `triton-xpu==3.7.1`, `transformers==5.8.0`, oneCCL 2021.15.9 |

The base image is **not usable as-is**. It pins `libze_intel_gpu 1.15.37833`
(ICR 26.14) + loader 1.28.2 + IGC 2.32.7, while the host's `xe` kernel driver pairs
with the 1.17 series. That UMD/KMD mismatch does not fail at init — the container
starts and answers `/health` — and faults later with `DEVICE_LOST` during
Inductor/Triton autotune and `OUT_OF_RESOURCES` inside the TP2 allreduce, which both
look like VRAM or oneCCL problems. `image/Dockerfile` overlays
`libze_intel_gpu 1.17.39395` / `libze_loader 1.32.0` / `libigc 2.40.13+0` from the
OMIX 0.4.0 tag and finishes with a `ctypes.CDLL` load check so a broken overlay
fails the build. Use the **Ubuntu** `-devel-ubuntu24.04` tag: the host's
Arch-built `libigc` needs `GLIBCXX_3.4.35`, which noble does not have.

Both `FROM` inputs are pinned **by digest**, not by tag — the two digests in the
table above are what `image/Dockerfile` pulls. A registry tag is mutable, and a
quietly-moved tag reproduces exactly the fault the overlay exists to fix, so a
tag-only pin would not make this repo reproducible. Override with `--build-arg`
only deliberately, then re-run `scripts/verify.sh`, which reads the engine's own
printed config rather than your env file. Rebuild after either side changes
(`docker build -t vllm-llmscaler:icr26.35 image/`).
Full write-up: [driver-overlay.md](driver-overlay.md).

## Model quantization

| item | value | why |
|---|---|---|
| target | `Qwen/Qwen3.8-27B-FP8` @ commit `017b9c7af6b5` (29 GB) | `quant_method: fp8`, `fmt: e4m3`, dynamic activation scheme. Smartest of the quant matrix on this box |
| served as | `/models/target` (read-only bind) | |
| drafter | `rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70` @ `94f8b9cc1657` (2.6 GB) | 1.36B / 5-layer DSpark draft model, fine-tuned on **FP8** hidden states |
| drafter weights | sha256 `b6c45ca365d9…2ad335cc` | checked by `download-models.sh`; if it differs, every recorded number is unattributable |

FP8 over INT4 here because the drafter was trained on FP8 hidden states. That
pairing was decided in the **eager, k=4** era — suite 15.67 (FP8) vs 13.56 (INT4)
tok/s, pos-0 acceptance 55-84% vs 45-61% — so the *ratio* is the finding; the
absolute numbers describe a different configuration. The cost of FP8 is KV headroom.
Compare like-for-like, both under `DTYPE=auto`: KV 9.97 GiB and 1.46x max concurrency
at full 262k context, versus INT4's 15.37 GiB / 2.25x. `[HW]` If you drop
`MAX_MODEL_LEN` (262144), KV stops binding.

On the shipped `float16` default the FP8 side is **10.59 GiB / 1.58x = 415,029
tokens** (`GPU KV cache size` and `Available KV cache memory` in the container log;
`kv_cache_max_concurrency` on `/metrics`). The INT4 figures predate the dtype change
and were never re-measured under it, so do not re-derive the FP8-vs-INT4 ratio from
the 1.58x. The FP8 gain is a **two-knob** delta (+30,073 over the 384,956 auto
baseline) — do not attribute it to one flag or add the per-knob numbers:
`scripts/serve.sh`'s attribution table records +12,660 for the ssm pin alone and
389,265 against a 384,956 control for `--dtype float16` alone, because the pool is
leftover memory over a page size that both dtypes move.

FP8 KV is **load-bearing, not an optimization**: a measured `--kv-cache-dtype auto`
(fp16) boot of this config fails at 262k, quoting 12.94 GiB needed against 10.58 GiB
of available KV memory on *that* boot and estimating a maximum context of 212,480
(`v1/core/kv_cache_utils.py:778`; all three figures are the engine's own words). Do not
relax `KV_CACHE_DTYPE` and expect the full context. Do not extrapolate the headroom
from the dtype width either: the pool also carries the GDN recurrent state, whose
contribution is what the ssm pin below moves.

`VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1` converts the ~29 GB of fp8 shards with CPU
offload; without it, weight conversion exhausts device memory during load.

## Cache quantization

| item | value | why |
|---|---|---|
| KV cache | `--kv-cache-dtype fp8` | the measured config, and **required** for the full 262k context (a measured fp16-KV boot caps out at 212,480). FP8 KV is the difference between 1.46x and (on INT4) 2.25x concurrency headroom at 262k — that pair is like-for-like, both under `DTYPE=auto`; the shipped `float16` default moves the FP8 side to 1.58x |
| KV offload | **off** (`KV_OFFLOAD_SIZE=0`, flag omitted) | the host tier is not a performance knob here and it is a crash vector at long context. It never serves a read — `kv_offload_cpu_cache_read_usage_perc` and the external prefix-cache hit rate both stay `0.0` across every boot measured — while consuming host RAM proportional to use and, under `--kv-offloading-size 50`, driving `MemAvailable` from ~80-88 GiB to ~8.8 GiB on a 188 GiB box. A single ~160k-token request then took the whole engine down. `0` **omits** the flag — passing a literal 0 is not the same thing. Read [kv-offload-defect.md](kv-offload-defect.md) before re-enabling |
| GDN recurrent state | `--mamba-ssm-cache-dtype float16` | this is a hybrid GDN/linear-attention arch (`Qwen3_5ForConditionalGeneration`, `linear_attention` on 3 of every 4 layers), so the recurrent state is a **second, separate** cache dtype and `auto` does not mean "inherit `--dtype`" — see `MAMBA_SSM_CACHE_DTYPE` below. Throughput-neutral on its own; it buys KV capacity and is what lets the ESIMD GDN core run alongside `--dtype float16` |
| engine dtype | `--dtype float16` | **the measured-best value and the shipped default.** `--dtype auto` resolves to bfloat16 for this FP8 checkpoint and measures **-6.0%** C1 / **-8.0%** conc=10 against it; see `DTYPE` below. This row used to say the opposite — "**do not force `--dtype float16`** — it downcasts the LM head and degrades output on this arch" — and was wrong on both halves. The LM head stays bf16 under *either* setting (`scripts/serve.sh`: "weights stay fp8 and lm_head stays bf16 in both arms"), so `--dtype` downcasts nothing here; and "degrades output" did not reproduce either — the quality battery matched the control (`serve.sh`: paired GSM8K 29/30 on both arms *and* control, 19/20 exact-answer with the **same** single miss, 0/20 non-reproducible greedy outputs, correctness 15/15), and the load-bearing acceptance counts are unmoved (C1 `26 169 7.500`; 512 accepted in both windows on either dtype). The soft NAT draft count reads 272 rather than 256 — read [recorded-metrics.md](recorded-metrics.md) and the header of `scripts/bench/acceptance.sh` before treating that as a result. The row predates the W57 paired A/B that settled it |

`int4_per_token_head` KV is not an option on XPU: it hits the 128 KiB local-data-
shared-memory wall in the attention kernel. `fp8` is the only quantized KV dtype
that works here.

## Patches

| patch | file | applied by |
|---|---|---|
| drafter architecture | [`../profiles/fp8/model-patches/drafter-config.json`](../profiles/fp8/model-patches/drafter-config.json) | `setup.sh` / `download-models.sh` |

That is the only patch. **No Python source patches, no bind-mounted `.py` files.**
Rationale: the drafter publishes `architectures: ["DSparkDraftModel"]` + `auto_map`,
neither of which resolves on 0.26 — the boot dies at `SpeculativeConfig` validation.
The patch sets `Qwen3DSparkModel` (the registered architecture) and drops `auto_map`
so the registered model is used instead of the shipped custom code. Nothing else in
the file may change; `setup.sh` diffs the live config against the expected one and
dies listing any differing keys. Upstream original kept for reference as
`drafter-config.upstream.json`. Full rationale:
[dspark-drafter.md](dspark-drafter.md).

The validated-but-rejected `cudagraph_utils.py` strict-stride patch is documented in
[cudagraph-stride-defect.md](cudagraph-stride-defect.md) and deliberately
not shipped: it costs 35.8% of batched throughput, and a bind-mounted `.py` can
silently stop applying after an image rebuild. Static `k=7` achieves correctness by
configuration instead.

## Runtime flags

Assembled by `scripts/serve.sh` (bind-mounted to `/serve.sh`); this is the literal
command the measured run executed:

```
vllm serve /models/target \
  --host 0.0.0.0 --port 8103 --served-model-name secondary \
  --tensor-parallel-size 2 --dtype float16 \
  --mamba-ssm-cache-dtype float16 \
  --max-model-len 262144 --max-num-seqs 10 --max-num-batched-tokens 4096 \
  --block-size 64 \
  --enable-prompt-tokens-details --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true, "preserve_thinking": true, "reasoning_effort": "xhigh"}' \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 --trust-remote-code --enable-prefix-caching \
  --speculative-config '{"method":"dspark","model":"/models/drafter","num_speculative_tokens":7}' \
  --compilation-config '{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":80}'
```

Flags worth explaining:

- `--compilation-config mode:none` + `cudagraph_mode:FULL_DECODE_ONLY` — graphs on,
  `torch.compile` off. Compiling the base model *does* make it faster in isolation,
  but it collapses DSpark acceptance to ~1.00, so every compiled variant loses once
  drafting is on. `mode:"none"` is what every llm-scaler Qwen/DFlash recipe uses.
- `max_cudagraph_capture_size: 80` — **not a tuning knob.** It must equal
  `MAX_NUM_SEQS x (1 + NUM_SPEC_TOKENS)` = `10 x 8`. Higher wastes capture memory on
  shapes that can never dispatch; lower silently drops the saturated batch to eager
  (-38%). `setup.sh` enforces the rule. `[HW]` Recompute it whenever you change
  `MAX_NUM_SEQS` or `k`.
- `--speculative-config method:dspark` — V2-runner-only, and it natively implements
  the anchor-is-first-prediction readout that other routes hand-patch into
  `dflash.py`. `method=dflash` **does not boot with this drafter on 0.26**
  (`DFlash` auto-prefixing produces an unregistered architecture).
- `num_speculative_tokens: 7` — static k. With graphs on, `k=4` corrupts output
  (6/15) while `k=1, 6, 7` are clean; there is no divisibility rule that explains
  the pattern, so k=7 is a validated pin, not an optimum. `k != 7` also costs ~30%
  of single-stream speed (k=6/k=4 C1 ~101-105 vs k=7 151).
- `--block-size 64` with `--max-num-batched-tokens 4096` — the measured pairing.
- `--dtype float16` + `--mamba-ssm-cache-dtype float16` — the shipped, measured-best
  permutation, and **two separate levers, not one**: `--dtype` buys essentially all
  the speed (+6.0% C1, +8.0% conc=10, C1 TTFT -19.6%), while the ssm pin is the
  capacity lever and is throughput-neutral by itself (+12,660 tokens alone, against
  +4,309 for `--dtype` alone, and +30,073 with both set — non-additively). Passing
  only `--dtype float16` boots and measures the same speed (243.15 vs 241.87 conc=10,
  inside spread) but forfeits the capacity, because the GDN ESIMD core's gate demands
  conv_state *and* ssm_state float16 — and `verify_and_update_config` has already
  rewritten an `auto` ssm dtype to the checkpoint's `float32`. Neither flag downcasts
  the LM head; it stays bf16 in every arm. The mechanism behind the `--dtype` win is
  **unresolved** — the two `float16`-gated ESIMD paths are correlates, falsified as
  the cause by ablation — so read the table in `scripts/serve.sh` before quoting a
  reason for it. `DTYPE=auto` + `MAMBA_SSM_CACHE_DTYPE=auto` is the documented control
  spelling that restores the pre-W57 bfloat16 baseline.
- `--tool-call-parser qwen3_xml --reasoning-parser qwen3` + the chat-template kwargs
  — the serving surface `verify.sh` checks. `enable_thinking`/`preserve_thinking`
  via `--default-chat-template-kwargs` is what the reference numbers were measured
  with; the suite harness overrides it off, which is also how the references were taken.
- `--trust-remote-code` — required by the Qwen3_5 architecture.

## Environment variables

Container env, set by the compose file from `../profiles/fp8/fp8.env` plus a fixed
block:

| variable | value | why |
|---|---|---|
| `VLLM_TARGET_DEVICE` | `xpu` | set by `serve.sh`; the image's entrypoint is replaced, so this is not inherited |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | fork + XPU contexts deadlocks |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | 262144 exceeds the arch default window |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | **mandatory**: dspark exists only in the V2 GPU model runner. `config/vllm.py` forces it when `method=dspark` but not otherwise, so pin it in comparison arms too or your A/B measures V2-vs-V1 |
| `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` | `1` | fp8 weight conversion without exhausting device memory |
| `VLLM_XPU_ENABLE_XPU_GRAPH` | `1` (via `XPU_GRAPH=1`) | the graph switch itself. Swept: neither this nor the two knobs below improve on baseline |
| `VLLM_XPU_USE_SAMPLER_KERNEL` | `1` (via `XPU_SAMPLER`) | historically bundled with the graph switch; separated only so a graph run can be attributed between capture and sampler |
| `ZE_FLAT_DEVICE_HIERARCHY` | `COMPOSITE` | swept; `FLAT` is no better |
| `TRITON_DEFAULT_BACKEND` | `intel` | keeps Triton off any accidentally-present CUDA backend |
| `CCL_SYCL_ALLREDUCE_TMP_BUF` | `1` | **the only mitigation found for the oneCCL IPC-handle wedge** (0/6 vs 4/5 same-run controls, p=0.0152). The wedge does not self-heal and `/health` stays 200 while it spins. Do not turn this off |
| `CCL_SYCL_ALLGATHERV_TMP_BUF` | `0` | hard gate on Battlemage: pinning the allgatherv buffer faults. Do not "harmonise" it with the line above |
| `CCL_TOPO_P2P_ACCESS` | `1` | P2P works here. `[HW]` See `CCL_NO_P2P` below before you copy someone else's no-P2P block |
| `CCL_ZE_IPC_EXCHANGE` | `pidfd` | matches what bare metal used |
| `CCL_ENABLE_SYCL_KERNELS` | `1` | collectives run on-device |
| `CCL_SYCL_ALLGATHERV_{SMALL,SCALEOUT}_THRESHOLD` | `131072` / `1048576` | measured tuning from the bare-metal recipe |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` | models are on disk; never let the container reach for the hub |
| `VLLM_CACHE_ROOT` / `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` | under `/mnt/cache/.cache` | Inductor/Triton caches fill a 56 GB boot drive during autotune. `[HW]` point at your big pool |
| `CCL_NO_P2P` | `0` | **kept only to reproduce a documented failure.** Set to 1 and this host cannot start at all: `ze_handle_manager.cpp:42 mem_to_ipc_handle: EXCEPTION: device_fd is invalid value`, under pidfd, sockets, FLAT, and COMPOSITE. The no-P2P requirement came from a PCIe topology this box does not share, and TP2 comm is already cheap here (TP1 no-spec 10.13 vs TP2 8.60 tok/s), bounding the whole lever at ~1.5 tok/s. `[HW]` This is the flag a different topology may genuinely need — but test it, do not assume |
| `FI_TCP_IFACE` / `CCL_ATL_TRANSPORT` / `CCL_ATL_SRC_IFACE` | unset on the P2P path | host interface pinning is irrelevant inside the container until `by-path` is filtered; `serve.sh` unsets these so a host export cannot leak in |

`serve.sh` also sources `/opt/intel/oneapi/setvars.sh` and the CCL `vars.sh` inside
the container. Nothing is loaded from `/opt/intel/oneapi` on the **host** — the venv
bundles it — so the oneAPI toolkit is not required on the host. (Caution: removing
the host toolkit also removes `level-zero-loader` and kills GPU init.)

## Serve knobs

| variable | value | note |
|---|---|---|
| `PORT` / `SERVED_NAME` | `8103` / `secondary` | `[HW]` free port. Keep the served name if anything upstream routes by model name |
| `TP_SIZE` | `2` | `[HW]` must match the number of B70s given to the container |
| `MAX_MODEL_LEN` | `262144` | `[HW]` lowering it frees KV headroom |
| `MAX_NUM_SEQS` | `10` | `[HW]` drives the graph ceiling. A saturated engine always sits at this cap, so benchmark **here**, not at conc=8 |
| `GPU_MEM_UTIL` | `0.92` | `[HW]` lower it if anything else shares the GPUs |
| `KV_OFFLOAD_SIZE` | `0` (off) | host KV tier in GB. `[HW]` **off by default and stays off**: the tier never serves a read here, and at `50` it consumed ~80 GiB of host `MemAvailable` and killed the engine on a single long request. `0` omits `--kv-offloading-size` entirely — a literal `0` is a different config from an absent flag. See [kv-offload-defect.md](kv-offload-defect.md) |
| `SPEC` / `SPEC_METHOD` | `1` / `dspark` | |
| `SPEC_TOKENS_SCHEDULE` | **unset** | `[never with graphs]` it garbles output (6-13/30 correct) with healthy-looking acceptance counters. `setup.sh` aborts if it is set |
| `WARMUP` | `1` | burns 64 decode steps through a throwaway completion after `/health`, because in eager mode Triton JITs during the first inferences and those first requests otherwise time out empty |
| `COMPILE` / `PREFIX_CACHE` / `XPU_GRAPH` | `1` / `1` / `1` | `COMPILE=0` adds `--enforce-eager`; see the compilation-config note above |
| `DTYPE` | `float16` | model/activation dtype, passed as `--dtype`. **The measured-best value, not vLLM's default**: `auto` resolves to bfloat16 for this FP8 checkpoint, and `float16` measured **+6.0%** C1 probe, **+6.8%** conc=1, **+11.2%** conc=5, **+8.0%** conc=10, with C1 TTFT 170.6 → 137.3 ms (−19.6%) — each several times the within-arm spread. Pass `DTYPE=auto` to reproduce the pre-W57 baseline. The **mechanism is unresolved**: the two `float16`-gated ESIMD paths (fused add-RMS-norm and the GDN speculative core) appear only under fp16, but switching either off — or both at once — leaves the win intact, so they are correlates rather than the cause. Read the ablation table in `scripts/serve.sh` before quoting any reason for it |
| `MAMBA_SSM_CACHE_DTYPE` | `float16` | GDN recurrent-state dtype, passed as `--mamba-ssm-cache-dtype` (omitted entirely when `auto`). `auto` does **not** mean "inherit from `DTYPE`": `_mamba_state_dtype` would inherit, but `Qwen3_5ForConditionalGenerationConfig.verify_and_update_config` (`model_executor/models/config.py:752-777`) rewrites `auto` to the checkpoint's `mamba_ssm_dtype` first, and that is `float32`. So at `auto` the recurrent state is fp32 whatever `DTYPE` says. Its own contribution is KV capacity (+12,660 tokens alone, +30,073 alongside `DTYPE=float16`; the two do **not** add — the pool is leftover memory over a page size both dtypes move) and it is throughput-neutral by itself. It is shipped as a default for that capacity, and because the GDN core needs it *together with* `DTYPE=float16` |

## Container and devices

| item | value | why |
|---|---|---|
| `network_mode` | `host` | vLLM binds `PORT` directly and oneCCL's bootstrap uses ephemeral ports between the two TP workers |
| `ipc` / `shm_size` | `host` / `32g` | TP workers share host memory for collectives |
| `devices` | the two B70 `card*` + `renderD*` only | `[HW]` resolved from `B70_BDFS`. The Arc A310 (`i915`) and the RTX 2000 Ada (`nvidia`) must stay out |
| `volumes` | generated `dri-by-path` -> `/dev/dri/by-path:ro` | **do not mount the host's directory.** oneCCL aborts in `fill_device_fds` on the dangling entries for GPUs the container cannot see. `setup.sh` generates exactly four symlinks from your BDFs and refuses to start if any dangles — [dri-by-path.md](dri-by-path.md) |
| model mounts | `:ro` | |

The measured host: CachyOS, kernel `7.2.7-1-cachyos`, `intel-compute-runtime
26.35.39758.10`, `level-zero-loader 1.32.0`, Docker 29.8.2, Compose 5.6.0, B70s at
`0000:03:00.0` and `0000:83:00.0` (both `8086:e223`, driver `xe`).

## INT4 profile (placeholder)

`../profiles/int4/` holds **no runnable config**, and `setup.sh --profile int4`
refuses rather than booting something unverified. The constraints that decide which
INT4 checkpoint and which `SPEC_METHOD` are even legal live in the single
authoritative copy at
[`../profiles/int4/README.md`](../profiles/int4/README.md) — checkpoint selection by
`quant_method` (not directory name; `auto-round` has no 0.26 registry entry),
`method=dspark` rather than `dflash`, `VLLM_USE_V2_MODEL_RUNNER=1` in every arm,
KV headroom vs acceptance, and re-deriving the ceiling before re-soaking.

## Re-verifying a deployment

```bash
./scripts/falsify/run-all.sh                 # prove the checks below can actually fail
./scripts/verify.sh --profile fp8            # attestation + 15 correctness + acceptance + 5 surface cells
./scripts/bench/run-all.sh --profile fp8     # all three metrics, idle engine
./scripts/soak.sh --cycles 6 --cycle-seconds 600
```

What each one proves: [verification.md](verification.md).

`falsify/run-all.sh` runs four harnesses — the attestation rules against a
synthetic engine log, the `setup.sh` preflight gates, the acceptance formula, and
the `conc.py` `--max-tokens` guard — and requires every one of them to fire while
the clean baseline stays green. Run it after editing any guard, and pass
`CONTAINER=<name>` to cross-check the attestation regexes against a real engine log
(vLLM has changed that repr before, and a regex that stops matching looks exactly
like a correctly failing check).

`verify.sh` reads the **engine's own printed config** from the container log, not
your env file, and checks that the capture-size ladder actually *contains* 80 —
echoing the ceiling number is not proof the 80-token batch is graphed.
