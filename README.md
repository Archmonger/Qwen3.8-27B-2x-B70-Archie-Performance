# Qwen3.8-27B on 2x Intel Arc Pro B70 — Archie Performance

Reproducible vLLM serving on **2x Intel Arc Pro B70 (TP=2, XPU)** with DSpark
speculative decoding and FULL decode graphs. This repo holds everything needed to
stand up a container byte-comparable to the configuration that produced the recorded
numbers below: the image recipe, the launcher, the per-profile config, the drafter
config patch, and the harnesses that re-measure and re-attest it.

- **FP8 profile: implemented, deployed, soak-passed.** `sudo ./setup.sh`
- **INT4 profile: placeholder.** No config, and no unbootable config either — only
  READMEs describing the shape it will take. See
  [profiles/int4/README.md](profiles/int4/README.md).

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
scripts/lib/                   profile-env parser + the attestation rules (shared)
scripts/falsify/run-all.sh     proves every guard and attestation rule CAN fail
scripts/download-models.sh     pinned `hf` fetch + drafter patch + weight sha256 check
scripts/bench/                 concurrency, C1/natural probes, realistic suite, surface
profiles/fp8/                  fp8.env + docker-compose.yml + model-patches/
profiles/int4/                 PLACEHOLDER — READMEs only, no config
data/                          realistic-suite-v1.json, prompt512.txt
docs/                          configuration, measurement notes, the stride defect,
                               the driver overlay, by-path, oneCCL/P2P, the drafter,
                               verification discipline
```

## Installation

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

To re-check a deployment that is already up — attestation, correctness, acceptance,
serving surface, every guard's ability to fail, and a sustained-load soak — use the
command block and explanation in
[docs/configuration.md#re-verifying-a-deployment](docs/configuration.md#re-verifying-a-deployment).

## Qwen3.8-27B-FP8 Recorded Performance Metrics

Target `Qwen/Qwen3.8-27B-FP8`, drafter `rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70`,
image `vllm-llmscaler:icr26.35` (vLLM `0.26.1.dev0+g568afb3a1.d20260907`),
TP=2, `MAX_NUM_SEQS=10`, 262144-token context, KV cache `fp8`, DSpark `k=7` static
with `FULL_DECODE_ONLY` graphs at ceiling 80. Full configuration:
[docs/configuration.md](docs/configuration.md). All numbers below are from the same
configuration, so they are comparable with each other.

Every cell is a measurement taken on this host with the configuration above. Where a
row carries several recordings they are separated by `·`; `/` separates probe windows
that were measured together. Values quoted as a range are repeated reps, not a
confidence interval, and the reps behind each row are not interchangeable protocols —
[docs/measurement-notes.md](docs/measurement-notes.md) says which recording came from
which harness and which are within-boot reps.

| metric | recorded | how |
|---|---|---|
| **conc=10 aggregate (the operating point)** | **227.13 · 229.59 · 230.47** | `conc.py --cells 10:30 --max-tokens 256` |
| conc=12 aggregate | 221.50 · 228.81 · 230.41 | `conc.py --cells 12:30 --max-tokens 256` |
| conc=9 aggregate | 212.42 · 206.01-209.50 | `conc.py --max-tokens 256` |
| conc=8 aggregate | 202.56 · 202.64-205.11 | `conc.py --max-tokens 256` |
| conc=1 aggregate | 42.6 · 43.95-43.97 | `conc.py --cells 1:6 --max-tokens 256` |
| C1 counting probe (`"1 2"`, 16 tok) | **151.12-151.98** (peak 152.65) | `probe.py --prompt "1 2" --max-tokens 16 --iters 10` |
| natural 512-token probe | **58.69-58.92** (peak 58.94) | `probe.py --max-tokens 96 --iters 6` |
| realistic suite `tok_s_1_100_after_ttft` | **40.21-41.58** median (3 runs) | `suite.py` |
| suite TTFT | 161-189 ms median | `suite.py` |
| acceptance length (C1 / natural) | **7.500** (26/169) / **3.000** (256/512) — identical on every recording | `acceptance.sh` |
| correctness | 15/15 · 30/30 per graphs-on A/B arm | `correctness.sh` |
| serving surface | 5/5 | `surface.py` |
| graph capture cost | 0.58-0.60 GiB/rank, 6 s | engine log |
| configured capture ladder | `[1,2,4,8,16,24,32,40,48,56,64,72,80]` (13 sizes to 80) | engine log (`cudagraph_capture_sizes`) |
| soak | 6/6 cycles, 3894 s, 0 wedges, 0 restarts | `soak.sh` |

`--max-tokens` is part of the metric, not a knob: `aggregate_tok_s` moves ~15%
between 96 and 256 at `conc=10`, which is what once made a correct engine look ~13%
depressed. What the spread actually is, which third-party numbers on this hardware are
**not** parity targets, and which figures from the investigation were eager-era and are
not comparable to anything above:
[docs/measurement-notes.md](docs/measurement-notes.md).

## Qwen3.8-27B-INT4 Recorded Performance Metrics

> **PLACEHOLDER — this repo holds no INT4 measurement.**
>
> Reserved for the INT4 profile's numbers, in the same shape and with the same metric
> names as the FP8 table above. The `how` column pins every parameter that moves the
> number, mirroring the FP8 canonical set so the two profiles stay comparable —
> measure at those values, or change the column and the cell together:
>
> | metric | value | how |
> |---|---|---|
> | conc=10 aggregate | *TBD* | `conc.py --cells 10:30 --max-tokens 256` |
> | conc=9 aggregate | *TBD* | `conc.py --max-tokens 256` |
> | conc=8 aggregate | *TBD* | `conc.py --max-tokens 256` |
> | C1 counting probe | *TBD* | `probe.py --prompt "1 2" --max-tokens 16 --iters 10` |
> | natural 512-token probe | *TBD* | `probe.py --max-tokens 96 --iters 6` |
> | realistic suite `tok_s_1_100_after_ttft` | *TBD* | `suite.py` |
> | acceptance length (C1 / natural) | *TBD* | `acceptance.sh` |
> | correctness / serving surface | *TBD* | `correctness.sh` / `surface.py` |
> | graph capture cost | *TBD* | engine log |
> | soak | *TBD* | `soak.sh` |
>
> Until those cells are filled from a run this repo produced, do not substitute
> numbers from elsewhere. Why the investigation's INT4 figures (C1 71.08, natural
> 21.19, no-spec 10.93 / 11.22) are not usable here:
> [docs/measurement-notes.md#the-int4-placeholder-cells](docs/measurement-notes.md#the-int4-placeholder-cells).
> The constraints that decide which INT4 checkpoint and which `SPEC_METHOD` are even
> legal: [profiles/int4/README.md](profiles/int4/README.md).

## Documentation

| doc | about |
|---|---|
| [docs/configuration.md](docs/configuration.md) | the whole measured config: pinned image + digests, model/KV quantization, patches, runtime flags, env vars, serve knobs, container and devices |
| [docs/measurement-notes.md](docs/measurement-notes.md) | how to read the metrics table: where each recording comes from, the `max_tokens` mismatch, spread, eras, and the third-party numbers that are not parity targets |
| [docs/verification.md](docs/verification.md) | what each check proves, and the harness bugs that misreported healthy engines |
| [docs/cudagraph-stride-defect.md](docs/cudagraph-stride-defect.md) | the graphs x DSpark defect, six falsified theories, the rejected patch, and the ceiling bug |
| [docs/driver-overlay.md](docs/driver-overlay.md) | why the image is rebuilt from the pinned base |
| [docs/dri-by-path.md](docs/dri-by-path.md) | the filtered DRM directory, and the two failure modes it prevents |
| [docs/oneccl-and-p2p.md](docs/oneccl-and-p2p.md) | the IPC-handle wedge, the one knob that mitigates it, and why no-P2P cannot start here |
| [docs/dspark-drafter.md](docs/dspark-drafter.md) | drafter pairing, the config patch, why `method=dflash` cannot boot |

## Provenance

Everything here was measured on one host (2x Arc Pro B70, TP2, CachyOS) and the
files are the ones that host runs. Numbers are that box's: the ceiling rule, the
correctness gates, and the harnesses travel; the absolute tok/s figures, the
oneCCL settings, and the P2P verdict may not.
