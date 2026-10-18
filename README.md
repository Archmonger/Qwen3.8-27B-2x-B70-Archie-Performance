# Qwen3.8-27B on 2x Intel Arc Pro B70 — Archie Performance

Reproducible vLLM serving on **2x Intel Arc Pro B70 (TP=2, XPU)** with DSpark
speculative decoding and FULL decode graphs. This repo holds everything needed to
stand up a container byte-comparable to the configuration that produced the recorded
numbers below: the image recipe, the launcher, the per-profile config, the drafter
config patch, and the harnesses that re-measure and re-attest it.

- **FP8 profile: implemented, deployed, soak-passed.** `sudo ./setup.sh --profile fp8`
- **INT4 profile: placeholder.** No config, and no unbootable config either — only
  READMEs describing the shape it will take, and
  `sudo ./setup.sh --profile int4` refuses with an explanation rather than booting
  something unverified. See [profiles/int4/README.md](profiles/int4/README.md).

Every configuration detail lives in
[docs/configuration.md](docs/configuration.md): the pinned image and its digests, the
drafter patch, every environment variable, every serve knob, and the container/device
layout, each with the reason it is set that way. Read
[docs/verification.md](docs/verification.md) before trusting any number in here, and
[docs/measurement-notes.md](docs/measurement-notes.md) before quoting one. The
graph-stride defect in [docs/cudagraph-stride-defect.md](docs/cudagraph-stride-defect.md)
corrupts output while throughput and acceptance counters both stay healthy, which is
why this repo ships a correctness probe and an attestation step rather than just a
compose file.

## Contents

```
setup.sh                       preflight -> build -> generate -> start -> verify
image/Dockerfile               pinned base + Intel GPU userspace driver overlay
scripts/serve.sh               in-container launcher (assembles the vllm serve line)
scripts/verify.sh              config attestation + correctness + acceptance + surface
scripts/soak.sh                sustained-load soak (oneCCL IPC-handle wedge)
scripts/lib/                   profile-env parser + the attestation, engagement and
                               degeneracy rules (shared with the harness)
scripts/falsify/run-all.sh     proves every guard and attestation rule CAN fail
scripts/download-models.sh     pinned `hf` fetch + drafter patch + weight sha256 check
scripts/bench/                 named-test concurrency sweep, C1/natural probes,
                               prompt processing + TTFT, realistic suite, surface
scripts/bench/arm.sh           one A/B arm: boot with overrides, attest, engage-assert,
                               then gate and measure through the harnesses above
scripts/instrument/            dispatch-census modules arm.sh mounts, so a knob can be
                               shown to engage instead of assumed to have
profiles/fp8/                  fp8.env + docker-compose.yml + model-patches/
profiles/int4/                 PLACEHOLDER — READMEs only, no config
data/                          realistic-suite-v1.json + the four test prompts
docs/                          configuration, the recorded metrics, measurement notes,
                               the stride defect, the driver overlay, by-path,
                               oneCCL/P2P, the drafter, verification discipline
```

The four test prompts are `prompt-planning.txt`, `prompt-counting.txt`,
`prompt-prose.txt`, and `prompt-coding.txt`. Each is bound to a generation budget in
the `TESTS` table of `scripts/bench/conc.py`, so a test name selects prompt and budget
together — see the metrics table below.

## Installation

Requirements: Linux with the `xe` kernel driver, two Arc Pro B70s, Docker with the
compose plugin, `python3` on the host, ~33 GB free for models and ~24 GB for
images. Root (or docker-group membership) for `setup.sh`.

```bash
git clone git@github.com:Archmonger/Qwen3.8-27B-2x-B70-Archie-Performance.git
cd Qwen3.8-27B-2x-B70-Archie-Performance

./scripts/download-models.sh --models-dir /mnt/cache/models   # 33 GB, public, pinned

# FP8 — the implemented profile: build + start + verify
sudo ./setup.sh --profile fp8

# INT4 — a placeholder: exits 3 with an explanation instead of starting a
# container, because profiles/int4/ holds no runnable config
sudo ./setup.sh --profile int4
```

`--profile` selects between the two paths above: `fp8` is both the argument and the
default, while `int4` has no version that boots — it is refused at profile dispatch,
before any build or start, so nothing unverified can come up from this repo.

Once the FP8 profile is up:

```bash
curl -s http://127.0.0.1:8103/v1/models
curl -s http://127.0.0.1:8103/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"secondary","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}'
```

Useful flags: `--profile` (`fp8` | `int4`), `--dry-run` (preflight + resolved plan,
changes nothing), `--rebuild`, `--no-start`, `--b70-bdfs`, `--models-dir`,
`--cache-dir`, `--run-dir`. `-h` for all.

`setup.sh` is idempotent and fails closed — it preflights the driver, PCI binding,
model weights, drafter config, graph ceiling and KV-offload hotfix before it will
start a container, and refuses rather than booting a configuration known to be
broken. What each check verifies and why it exists is in
[verification.md#what-setupsh-refuses-to-boot](docs/verification.md#what-setupsh-refuses-to-boot).

To re-check a deployment that is already up — attestation, correctness, acceptance,
serving surface, every guard's ability to fail, and a sustained-load soak — use the
command block and explanation in
[docs/configuration.md#re-verifying-a-deployment](docs/configuration.md#re-verifying-a-deployment).

## Qwen3.8-27B-FP8 Recorded Performance Metrics

> [!NOTE]
> Target `Qwen/Qwen3.8-27B-FP8`, drafter `rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70`, docker image `vllm-llmscaler:icr26.35` (vLLM `0.26.1.dev0+g568afb3a1.d20260907`), values are **median of 3 reps**.

> [!WARNING]
> Intel Arc Pro B70 [does **not** support](https://www.techpowerup.com/gpu-specs/arc-pro-b70.c4388) hardware native FP8 vector **or** matrix operations. As a result, this image has an upper bound on maximum performance. Despite this, my recipe has the fastest B70 8-bit metrics I've seen. Additionally, FP8 is the only low precision format provided directly by Alibaba.

| Test Name | Concurrency | Tokens Generated | Value | Command |
|---|---:|---:|---:|---|
| Technical Planning | 10 | 2048 | **297.95** | `conc.py --test planning --cells 10:30` |
| Technical Planning | 5 | 2048 | 200.88 | `conc.py --test planning --cells 5:20` |
| Technical Planning | 1 | 2048 | 51.11 | `conc.py --test planning --cells 1:6` |
| Counting | 10 | 512 | **732.91** | `conc.py --test counting --cells 10:30` |
| Counting | 5 | 512 | 519.63 | `conc.py --test counting --cells 5:20` |
| Counting | 1 | 512 | 143.34 | `conc.py --test counting --cells 1:6` |
| Prose | 10 | 2048 | **233.30** | `conc.py --test prose --cells 10:30` |
| Prose | 5 | 2048 | 156.82 | `conc.py --test prose --cells 5:20` |
| Prose | 1 | 2048 | 44.60 | `conc.py --test prose --cells 1:6` |
| Coding | 10 | 2048 | **485.90** | `conc.py --test coding --cells 10:30` |
| Coding | 5 | 2048 | 325.41 | `conc.py --test coding --cells 5:20` |
| Coding | 1 | 2048 | 89.29 | `conc.py --test coding --cells 1:6` |
| Prompt Processing | 1 | — | **3 207** tok/s | `prefill.py --length-sweep` |
| TTFT | 1 | — | **188.1** ms | `prefill.py --test …` (median over the four test prompts) |

## Qwen3.8-27B-INT4 Recorded Performance Metrics

> [!NOTE]
> **PLACEHOLDER — this recipe is in-progress.**
>
> Reserved for the INT4 profile's numbers, in the same shape and with the same metric
> names as the FP8 table above. The `how` column pins every parameter that moves the
> number, mirroring the FP8 canonical set so the two profiles stay comparable —
> measure at those values, or change the column and the cell together:

> [!WARNING]
> This recipe is W4A16, which has higher theoretical accuracy compared to W4A4, but comes with a performance penalty caused by the need to dequantize from 4 bit to 16 bit. Despite this, this recipe has fantastic speed metrics due to the B70 platform having support for 16 bit vector **and** and matrix operations.

> | Test Name | Concurrency | Tokens Generated | Value | Command |
> |---|---:|---:|---:|---|
> | Technical Planning | 10 | 2048 | *TBD* | `conc.py --test planning --cells 10:30` |
> | Technical Planning | 5 | 2048 | *TBD* | `conc.py --test planning --cells 5:20` |
> | Technical Planning | 1 | 2048 | *TBD* | `conc.py --test planning --cells 1:6` |
> | Counting | 10 | 512 | *TBD* | `conc.py --test counting --cells 10:30` |
> | Counting | 5 | 512 | *TBD* | `conc.py --test counting --cells 5:20` |
> | Counting | 1 | 512 | *TBD* | `conc.py --test counting --cells 1:6` |
> | Prose | 10 | 2048 | *TBD* | `conc.py --test prose --cells 10:30` |
> | Prose | 5 | 2048 | *TBD* | `conc.py --test prose --cells 5:20` |
> | Prose | 1 | 2048 | *TBD* | `conc.py --test prose --cells 1:6` |
> | Coding | 10 | 2048 | *TBD* | `conc.py --test coding --cells 10:30` |
> | Coding | 5 | 2048 | *TBD* | `conc.py --test coding --cells 5:20` |
> | Coding | 1 | 2048 | *TBD* | `conc.py --test coding --cells 1:6` |
> | Prompt Processing | 1 | — | *TBD* | `prefill.py --length-sweep` |
> | TTFT | 1 | — | *TBD* | `prefill.py --test …` (median over the four test prompts) |

## Documentation

| doc | about |
|---|---|
| [docs/configuration.md](docs/configuration.md) | the whole measured config: pinned image + digests, model/KV quantization, patches, runtime flags, env vars, serve knobs, container and devices |
| [docs/recorded-metrics.md](docs/recorded-metrics.md) | every measurement the README does not show: acceptance, correctness, surface, graph capture, soak, the single-stream probes, the realistic suite |
| [docs/measurement-notes.md](docs/measurement-notes.md) | how to read the metrics table: where each recording comes from, why the prefill rate is a slope, the `max_tokens` mismatch, spread, eras, and the third-party numbers that are not parity targets |
| [docs/verification.md](docs/verification.md) | what each check proves, and the harness bugs that misreported healthy engines |
| [docs/cudagraph-stride-defect.md](docs/cudagraph-stride-defect.md) | the graphs x DSpark defect, six falsified theories, the rejected patch, and the ceiling bug |
| [docs/kv-offload-defect.md](docs/kv-offload-defect.md) | the long-context engine death and the ported vllm#52923 hotfix that fixes it, why the host KV tier ships **off** everywhere in this repo and how a single box opts in, and the falsified VRAM/ECC detours |
| [docs/driver-overlay.md](docs/driver-overlay.md) | why the image is rebuilt from the pinned base |
| [docs/dri-by-path.md](docs/dri-by-path.md) | the filtered DRM directory, and the two failure modes it prevents |
| [docs/oneccl-and-p2p.md](docs/oneccl-and-p2p.md) | the IPC-handle wedge, the one knob that mitigates it, and why no-P2P cannot start here |
| [docs/dspark-drafter.md](docs/dspark-drafter.md) | drafter pairing, the config patch, why `method=dflash` cannot boot |

## Provenance

Everything here was measured on one host (2x Arc Pro B70, TP2, CachyOS, PCIE 4.0 16x) and the
files are the ones that host runs. Numbers are that box's: the ceiling rule, the
correctness gates, and the harnesses travel; the absolute tok/s figures, the
oneCCL settings, and the P2P verdict may not.
