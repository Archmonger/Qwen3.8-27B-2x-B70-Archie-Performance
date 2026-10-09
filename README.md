# Qwen3.8-27B on 2x Intel Arc Pro B70 — Archie Performance

Reproducible vLLM serving on **2x Intel Arc Pro B70 (TP=2, XPU)** with DSpark
speculative decoding and FULL decode graphs. This repo holds everything needed to
stand up a container byte-comparable to the run that produced the recorded numbers
below: the image recipe, the launcher, the per-profile config, the drafter config
patch, and the harnesses that re-measure and re-attest it.

- **FP8 profile: implemented, deployed, soak-passed.** `sudo ./setup.sh`
- **INT4 profile: placeholder.** No config, and no unbootable config either — only
  READMEs describing the shape it will take. See
  [Qwen3.8-27B-INT4 Configuration (placeholder)](#qwen38-27b-int4-configuration-placeholder).

**Read [docs/verification.md](docs/verification.md) before trusting any number in
here.** One of the defects below is undetectable by throughput and acceptance
metrics together, which is why this repo ships a correctness probe and an
attestation step rather than just a compose file.

## Contents

```
setup.sh                       preflight -> build -> generate -> start -> verify
image/Dockerfile               pinned base + Intel GPU userspace driver overlay
scripts/serve.sh               in-container launcher (assembles the vllm serve line)
scripts/verify.sh              config attestation + correctness + acceptance + surface
scripts/soak.sh                sustained-load soak (oneCCL IPC-handle wedge)
scripts/lib/                   profile-env parser + the attestation rules (shared)
scripts/falsify/run-all.sh     proves every guard and attestation rule CAN fail
scripts/download-models.sh     pinned `hf` fetch + drafter patch + weight sha256 check
scripts/bench/                 concurrency, C1/natural probes, realistic suite, surface
profiles/fp8/                  fp8.env + docker-compose.yml + model-patches/
profiles/int4/                 PLACEHOLDER — READMEs only, no config
data/                          realistic-suite-v1.json, prompt512.txt
docs/                          the stride defect, the driver overlay, by-path,
                               oneCCL/P2P, the drafter, verification discipline
```

## Quick start

Requirements: Linux with the `xe` kernel driver, two Arc Pro B70s, Docker with the
compose plugin, `python3` on the host, ~33 GB free for models and ~24 GB for
images. Root (or docker-group membership) for `setup.sh`.

```bash
git clone git@github.com:Archmonger/Qwen3.8-27B-2x-B70-Archie-Performance.git
cd Qwen3.8-27B-2x-B70-Archie-Performance

./scripts/download-models.sh --models-dir /mnt/cache/models   # 33 GB, public, pinned
sudo ./setup.sh                                               # build + start + verify
```

Then:

```bash
curl -s http://127.0.0.1:8103/v1/models
curl -s http://127.0.0.1:8103/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"secondary","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}'
```

Useful flags: `--dry-run` (preflight + resolved plan, changes nothing),
`--rebuild`, `--no-start`, `--b70-bdfs`, `--models-dir`, `--cache-dir`,
`--run-dir`. `-h` for all.

`setup.sh` is idempotent and fails closed: it refuses to start the container if the
graph ceiling does not equal `MAX_NUM_SEQS x (1 + NUM_SPEC_TOKENS)`, if a dynamic
speculative schedule is set, if the drafter config is the unpatched upstream
variant, or if either PCI device is not bound to `xe`.

---

## Qwen3.8-27B-FP8 Recorded Performance Metrics

Target `Qwen/Qwen3.8-27B-FP8`, drafter `rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70`,
image `vllm-llmscaler:icr26.35` (vLLM `0.26.1.dev0+g568afb3a1.d20260907`),
TP=2, `MAX_NUM_SEQS=10`, 262144-token context, KV cache `fp8`, DSpark `k=7` static
with `FULL_DECODE_ONLY` graphs at ceiling 80. All numbers below are from the same
configuration, so they are comparable with each other.

### Headline

Two columns, because they mean different things. **Recorded** is the paired A/B
(2 reps per arm, alternating boot order) that established the configuration.
**Reproduced** is this repo's own harnesses run against the live engine afterwards
by whoever set this repo up — mostly single runs — so they exist to show the
numbers travel with the repo, not to re-litigate the A/B. Read the note under the
table before quoting a batched row.

Every row was produced with its harness's canonical parameters, which are now
part of the invocation rather than an implicit default — see the note under the
table. Use `scripts/bench/run-all.sh` to reproduce the whole table in one go.

| metric | recorded | reproduced | how |
|---|---|---|---|
| **conc=10 aggregate (the operating point)** | **227.13** | **229.59 / 230.47** | `conc.py --cells 10:30 --max-tokens 256` |
| conc=12 aggregate | 221.50 | 228.81 / 230.41 | `conc.py --cells 12:30 --max-tokens 256` |
| conc=9 / conc=8 aggregate | 212.42 / 202.56 | 206.01-209.50 / 202.64-205.11 | `conc.py --max-tokens 256` |
| conc=1 aggregate | 42.6 | 43.95-43.97 | `conc.py --cells 1:6 --max-tokens 256` |
| C1 counting probe (`"1 2"`, 16 tok) | ~152 | **151.12-151.98** (peak 152.65) | `probe.py --prompt "1 2" --max-tokens 16 --iters 10` |
| natural 512-token probe | ~59 | **58.69-58.92** (peak 58.94) | `probe.py --max-tokens 96 --iters 6` |
| realistic suite `tok_s_1_100_after_ttft` | — | **40.21-41.58** median (3 runs) | `suite.py` |
| suite TTFT | — | 161-189 ms median | `suite.py` |
| acceptance length (C1 / natural) | **7.500** (26/169) / **3.000** (256/512) | **7.500 (26/169) / 3.000 (256/512)** — exact | `acceptance.sh` |
| correctness | 30/30 per arm | 15/15 | `correctness.sh` |
| serving surface | 5/5 | 5/5 | `surface.py` |
| graph capture cost | 0.58 GiB/rank, ~6 s | 0.58-0.60 GiB/rank, 6 s | engine log |
| configured capture ladder | 13 sizes to 80 | `[1,2,4,8,16,24,32,40,48,56,64,72,80]` | engine log (`cudagraph_capture_sizes`) |
| soak | 6/6 cycles, 3894 s, 0 wedges, 0 restarts | — | `soak.sh` |

**`--max-tokens` is part of the metric, not a knob — and an earlier revision of
this table got it wrong.** `conc.py`'s `aggregate_tok_s` divides counted
`completion_tokens` by wall time, but wall time also pays for prefill. The prompt
here is a fixed ~243 tokens, so that uncounted prefill cost is amortised over
`max_tokens`: at `conc=10`, 96 tokens/request reads ~13% below 256 on the *same
boot*, while `conc=1` moves only ~4% because prefix caching makes prefill nearly
free for a single stream. That asymmetry is what made it look like a host problem.

The table above originally reported the batched cells at **197** against a
recorded **227.13** and called the gap an unidentified ~13% host-side depression.
That diagnosis was wrong, and the numbers in this paragraph are what caused it:
the verification run invoked `conc.py` directly, which fell through to a
`--max-tokens` **default of 96**, while the recorded A/B and both scripted callers
(`run-all.sh`, `soak.sh`) used **256**. Re-running one boot with alternating arm
order — the only change being the flag:

| `--max-tokens` | 96 | 256 | recorded |
|---|---|---|---|
| conc=10 | 193.23-202.94 | **229.16-231.64** | 227.13 |
| conc=1 | 42.37 | 44.32 | 42.6 |

The same boot reproduces the recorded column at 256. Nothing was wrong with the
engine, the image, or the host: 0 preemptions, 0 KV usage, 0 restarts, correct
graph capture throughout. `conc.py` now defaults to 256, prints `max_tokens` with
every cell, and refuses a non-canonical value unless you pass
`--confirm-noncanonical`, so this comparison can fail loudly instead of quietly
measuring a different metric. The natural-probe row had the same failure mode
(quoted 50.21 from a `--max-tokens 512` run against a recorded ~59 taken at 96)
and is now quoted at the canonical 96.

**A metric that moves ~15% on an unrecorded parameter is not a metric.** Quote the
harness, the cells, *and* `max_tokens` with any aggregate number.

Dead end, kept so nobody re-walks it: an orphaned root-owned `intel_gpu_top`
holding the GPU perf/OA unit open was blamed for the depression, then cleared —
it started Oct 3, so it was also running during the Oct 7 run that recorded
227.13 and cannot explain a delta between the two. The real cause was in the
harness all along.

The acceptance windows reproducing to the exact draft and accepted counts
(`C1 26 169 7.500`, `NAT 256 512 3.000`) is the strongest signal that this repo
boots the same engine: those counters are a fingerprint of the drafter, the target,
and k together.

**One correction to a figure that circulated earlier.** The realistic-suite metric
for this configuration is **~40-42 tok/s**, not the 15.67 that appears in older
notes. 15.67 was measured in the **eager, k=4** era, before graphs were enabled at
all, and it is not comparable to anything above. Likewise the "~1.6x from spec
decode" figure (13.76 DSpark k=4 vs 8.60 no-spec on `vllm bench serve`) and the
spec-off suite baseline of 10.32 are eager-era, k=4 measurements — useful as a
same-era A/B, useless as context for the graphed config. Quote the era with the
number.

### The two measurements that changed the number

1. **`MAX_NUM_SEQS x (1+k)` is the graph ceiling, and 72 was wrong.** A decode step
   whose token count exceeds `max_cudagraph_capture_size` silently falls back to
   eager. With `MAX_NUM_SEQS=10`, the saturated batch is `10 x 8 = 80` tokens, so at
   ceiling 72 the only state a busy engine ever reaches was ungraphed. Paired A/B
   (2 reps/arm, alternating boot order): conc10 **164.70 -> 227.13 (+37.9%)**,
   conc12 167.56 -> 221.50 (+32.2%), conc8/conc9 unchanged. Details:
   [docs/cudagraph-stride-defect.md](docs/cudagraph-stride-defect.md).
2. **Graphs + DSpark is safe only at a consistent stride.** Static `k=7` with
   `FULL_DECODE_ONLY` is 30/30 correct; the dynamic schedule that looked faster
   produced 6-13/30 correct while acceptance counters stayed healthy. Same document.

### Spread and what is not comparable

Run-to-run spread is real and metric-dependent, but much smaller than this README
previously claimed. With every harness at its canonical parameter, repeated reps
**on one boot** gave C1 **151.12-151.98** (peak 152.65), the natural 512-token probe
**58.69-58.92**, and conc=10 at `--max-tokens 256` **229.16-231.64** — a ~1% range.

That is reps-on-one-boot, which is *not* a session-to-session noise floor and must
not be quoted as one: it holds the boot, the box state and the cache state fixed, and
this host is documented to drift ~5% per 10 boots. Measure the floor for the
comparison you are actually making, by booting the identical config as a calibration
arm. Anything under ~3 tok/s on the natural metric needs repeats, and real A/Bs are
paired with alternating arm order rather than run once each.

Two numbers in this README's own history were passed off as spread and were not
spread at all: the ~13% batched gap (a `max_tokens` mismatch, above) and the
**47.70-55.44 C1 / 21.37-24.85 natural** range quoted by an earlier revision, which
is **eager-era**, k=4, graphs off — the era-mixing that "One correction to a figure
that circulated earlier" warns about, reproduced inside this README's own text.
Before attributing a gap to noise, check that both sides used the same `max_tokens`,
the same cells, the same probe, and the same era.

Two published numbers on this hardware are **not** parity targets and are quoted
here only so nobody re-derives them as a regression:

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

Always state which metric you are quoting. `tok_s_1_100_after_ttft` (`suite.py`),
`aggregate_tok_s` (`conc.py`), `median_tok_s_after_ttft` (`probe.py`), and the
engine log's "Avg generation throughput" are four different measurements.

---

## Qwen3.8-27B-FP8 Configuration

Everything below is what the measured run used. `[HW]` marks an item someone on a
slightly different box may plausibly have to change.

### Pinned image

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
Full write-up: [docs/driver-overlay.md](docs/driver-overlay.md).

### Model quantization

| item | value | why |
|---|---|---|
| target | `Qwen/Qwen3.8-27B-FP8` @ commit `017b9c7af6b5` (29 GB) | `quant_method: fp8`, `fmt: e4m3`, dynamic activation scheme. Smartest of the quant matrix on this box |
| served as | `/models/target` (read-only bind) | |
| drafter | `rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70` @ `94f8b9cc1657` (2.6 GB) | 1.36B / 5-layer DSpark draft model, fine-tuned on **FP8** hidden states |
| drafter weights | sha256 `b6c45ca365d9…2ad335cc` | checked by `download-models.sh`; if it differs, every number above is unattributable |

FP8 over INT4 here because the drafter was trained on FP8 hidden states. That
pairing was decided in the **eager, k=4** era — suite 15.67 (FP8) vs 13.56 (INT4)
tok/s, pos-0 acceptance 55-84% vs 45-61% — so the *ratio* is the finding; the
absolute numbers describe a different configuration. The cost of FP8 is KV headroom:
KV 9.97 GiB and 1.46x max concurrency at full 262k context, versus INT4's
15.37 GiB / 2.25x. `[HW]` If you drop `MAX_MODEL_LEN` (262144), KV stops binding.

`VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1` converts the ~29 GB of fp8 shards with CPU
offload; without it, weight conversion exhausts device memory during load.

### Cache quantization

| item | value | why |
|---|---|---|
| KV cache | `--kv-cache-dtype fp8` | the measured config; FP8 KV is the difference between 1.46x and (on INT4) 2.25x concurrency headroom at 262k |
| KV offload | `--kv-offloading-size 50` (GB) | host-memory tier. `0` **omits** the flag — passing a literal 0 is not the same thing |
| mamba/SSM cache | default (`--dtype auto`) | this is a hybrid GDN/linear-attention arch (`Qwen3_5ForConditionalGeneration`, `linear_attention` on 3 of every 4 layers) |
| engine dtype | `--dtype auto` | **do not force `--dtype float16`** — it downcasts the LM head and degrades output on this arch |

`int4_per_token_head` KV is not an option on XPU: it hits the 128 KiB local-data-
shared-memory wall in the attention kernel. `fp8` is the only quantized KV dtype
that works here.

### Patches

| patch | file | applied by |
|---|---|---|
| drafter architecture | [`profiles/fp8/model-patches/drafter-config.json`](profiles/fp8/model-patches/drafter-config.json) | `setup.sh` / `download-models.sh` |

That is the only patch. **No Python source patches, no bind-mounted `.py` files.**
Rationale: the drafter publishes `architectures: ["DSparkDraftModel"]` + `auto_map`,
neither of which resolves on 0.26 — the boot dies at `SpeculativeConfig` validation.
The patch sets `Qwen3DSparkModel` (the registered architecture) and drops `auto_map`
so the registered model is used instead of the shipped custom code. Nothing else in
the file may change; `setup.sh` diffs the live config against the expected one and
dies listing any differing keys. Upstream original kept for reference as
`drafter-config.upstream.json`. Full rationale:
[docs/dspark-drafter.md](docs/dspark-drafter.md).

The validated-but-rejected `cudagraph_utils.py` strict-stride patch is documented in
[docs/cudagraph-stride-defect.md](docs/cudagraph-stride-defect.md) and deliberately
not shipped: it costs 35.8% of batched throughput, and a bind-mounted `.py` can
silently stop applying after an image rebuild. Static `k=7` achieves correctness by
configuration instead.

### Runtime flags

Assembled by `scripts/serve.sh` (bind-mounted to `/serve.sh`); this is the literal
command the measured run executed:

```
vllm serve /models/target \
  --host 0.0.0.0 --port 8103 --served-model-name secondary \
  --tensor-parallel-size 2 --dtype auto \
  --max-model-len 262144 --max-num-seqs 10 --max-num-batched-tokens 4096 \
  --block-size 64 \
  --enable-prompt-tokens-details --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --kv-offloading-size 50 \
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
- `--tool-call-parser qwen3_xml --reasoning-parser qwen3` + the chat-template kwargs
  — the serving surface `verify.sh` checks. `enable_thinking`/`preserve_thinking`
  via `--default-chat-template-kwargs` is what the reference numbers were measured
  with; the suite harness overrides it off, which is also how the references were taken.
- `--trust-remote-code` — required by the Qwen3_5 architecture.

### Environment variables

Container env, set by the compose file from `profiles/fp8/fp8.env` plus a fixed
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

### Serve knobs

| variable | value | note |
|---|---|---|
| `PORT` / `SERVED_NAME` | `8103` / `secondary` | `[HW]` free port. Keep the served name if anything upstream routes by model name |
| `TP_SIZE` | `2` | `[HW]` must match the number of B70s given to the container |
| `MAX_MODEL_LEN` | `262144` | `[HW]` lowering it frees KV headroom |
| `MAX_NUM_SEQS` | `10` | `[HW]` drives the graph ceiling. A saturated engine always sits at this cap, so benchmark **here**, not at conc=8 |
| `GPU_MEM_UTIL` | `0.92` | `[HW]` lower it if anything else shares the GPUs |
| `SPEC` / `SPEC_METHOD` | `1` / `dspark` | |
| `SPEC_TOKENS_SCHEDULE` | **unset** | `[never with graphs]` it garbles output (6-13/30 correct) with healthy-looking acceptance counters. `setup.sh` aborts if it is set |
| `WARMUP` | `1` | burns 64 decode steps through a throwaway completion after `/health`, because in eager mode Triton JITs during the first inferences and those first requests otherwise time out empty |
| `COMPILE` / `PREFIX_CACHE` / `XPU_GRAPH` | `1` / `1` / `1` | `COMPILE=0` adds `--enforce-eager`; see the compilation-config note above |

### Container and devices

| item | value | why |
|---|---|---|
| `network_mode` | `host` | vLLM binds `PORT` directly and oneCCL's bootstrap uses ephemeral ports between the two TP workers |
| `ipc` / `shm_size` | `host` / `32g` | TP workers share host memory for collectives |
| `devices` | the two B70 `card*` + `renderD*` only | `[HW]` resolved from `B70_BDFS`. The Arc A310 (`i915`) and the RTX 2000 Ada (`nvidia`) must stay out |
| `volumes` | generated `dri-by-path` -> `/dev/dri/by-path:ro` | **do not mount the host's directory.** oneCCL aborts in `fill_device_fds` on the dangling entries for GPUs the container cannot see. `setup.sh` generates exactly four symlinks from your BDFs and refuses to start if any dangles — [docs/dri-by-path.md](docs/dri-by-path.md) |
| model mounts | `:ro` | |

The measured host: CachyOS, kernel `7.2.7-1-cachyos`, `intel-compute-runtime
26.35.39758.10`, `level-zero-loader 1.32.0`, Docker 29.8.2, Compose 5.6.0, B70s at
`0000:03:00.0` and `0000:83:00.0` (both `8086:e223`, driver `xe`).

### Re-verifying a deployment

```bash
./scripts/falsify/run-all.sh                 # prove the checks below can actually fail
./scripts/verify.sh --profile fp8            # attestation + 15 correctness + acceptance + 5 surface cells
./scripts/bench/run-all.sh --profile fp8     # all three metrics, idle engine
./scripts/soak.sh --cycles 6 --cycle-seconds 600
```

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

---

## Qwen3.8-27B-INT4 Recorded Performance Metrics

> **PLACEHOLDER — no INT4 result is published from this repo yet.**
>
> This section is reserved for the INT4 profile's numbers, in the same shape and
> with the same metric names as the FP8 section above:
>
> | metric | value | how |
> |---|---|---|
> | conc=10 aggregate | *TBD* | `conc.py --cells 10:30` |
> | conc=9 / conc=8 aggregate | *TBD* | `conc.py` |
> | C1 counting probe | *TBD* | `probe.py --prompt "1 2"` |
> | natural 512-token probe | *TBD* | `probe.py` |
> | realistic suite `tok_s_1_100_after_ttft` | *TBD* | `suite.py` |
> | acceptance length (C1 / natural) | *TBD* | `acceptance.sh` |
> | correctness / serving surface | *TBD* | `correctness.sh` / `surface.py` |
> | graph capture cost | *TBD* | engine log |
> | soak | *TBD* | `soak.sh` |
>
> Until those cells are filled from a run this repo produced, do not substitute
> numbers from elsewhere: the INT4 figures recorded during the investigation
> (C1 71.08, natural 21.19 with DSpark `k=7`, no-spec 10.93 / 11.22, KV
> 19.7 GiB, 35/35 correct) were measured on the base image with a **dynamic**
> schedule and **no graphs**, which is not a configuration this repo can currently
> stand up. See the placeholder notes below for the two traps that decide which
> INT4 checkpoint and which `SPEC_METHOD` are even legal.

## Qwen3.8-27B-INT4 Configuration (placeholder)

`profiles/int4/` holds **no runnable config** — only the READMEs that describe the
shape it will take — so nothing unverified can be booted.
`setup.sh --profile int4` refuses with an explanation rather than starting
something. The layout will mirror the FP8 profile:

```
profiles/int4/                       # TODO: to be added
  int4.env                           # TODO  image, quant, k, ceiling, KV dtype, CCL block
  docker-compose.yml                 # TODO  mirrors profiles/fp8/docker-compose.yml
  model-patches/
    drafter-config.json              # TODO  expected to be identical to the FP8 profile's
```

TODO items when this profile is implemented:

- **Pick the checkpoint by what the engine can load, not by the directory name.**
  `quant_method` is the contract. One INT4 checkpoint in this family declares
  `auto-round`, which has no entry in 0.26's quantization registry
  (`get_quantization_config("auto_round")` raises `ValueError: Invalid
  quantization method: auto-round`), while `gptq` / `gptq_marlin` / `auto_gptq` all
  resolve. Verify by running each `config.json` through vLLM's own resolver, not by
  reading the file.
- **`method=dflash` is unbootable with the published drafter on 0.26** (auto-prefix
  -> unregistered `DFlashQwen3DSparkModel`). Use `method=dspark`.
- **The same drafter is legitimate for INT4.** Its contract with the target is the
  hidden-state taps (`target_layer_ids [4,16,28,40,52]` of 64, `hidden_size` 5120,
  `vocab_size` 248320), which INT4 and FP8 text configs match exactly. C1 acceptance
  was identical (7.500) across both targets.
- **INT4 buys KV headroom, not automatically speed:** 19.7 GiB vs FP8's 9.97, but
  the drafter was fine-tuned on FP8 hidden states, so acceptance drops.
- **`VLLM_USE_V2_MODEL_RUNNER=1` in every arm, including no-spec**, or the A/B
  measures V1-vs-V2 as much as spec-vs-none.
- **Re-derive the graph ceiling and re-soak.** `MAX_NUM_SEQS x (1+k)` still applies,
  and the oneCCL wedge is a TP2 property of the platform, not of the quantization.
- Pin the image with `IMAGE=vllm-llmscaler-int4:icr26.35` if a second image is ever
  needed, so an FP8 and an INT4 container can never silently share a tag.

## Documentation

| doc | about |
|---|---|
| [docs/cudagraph-stride-defect.md](docs/cudagraph-stride-defect.md) | the graphs x DSpark defect, six falsified theories, the rejected patch, and the ceiling bug |
| [docs/driver-overlay.md](docs/driver-overlay.md) | why the image is rebuilt from the pinned base |
| [docs/dri-by-path.md](docs/dri-by-path.md) | the filtered DRM directory, and the two failure modes it prevents |
| [docs/oneccl-and-p2p.md](docs/oneccl-and-p2p.md) | the IPC-handle wedge, the one knob that mitigates it, and why no-P2P cannot start here |
| [docs/dspark-drafter.md](docs/dspark-drafter.md) | drafter pairing, the config patch, why `method=dflash` cannot boot |
| [docs/verification.md](docs/verification.md) | what each check proves, and the harness bugs that misreported healthy engines |

## Provenance

Everything here was measured on one host (2x Arc Pro B70, TP2, CachyOS) and the
files are the ones that host runs. Numbers are that box's: the ceiling rule, the
correctness gates, and the harnesses travel; the absolute tok/s figures, the
oneCCL settings, and the P2P verdict may not.
