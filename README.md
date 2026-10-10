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
scripts/lib/                   profile-env parser + the attestation rules (shared)
scripts/falsify/run-all.sh     proves every guard and attestation rule CAN fail
scripts/download-models.sh     pinned `hf` fetch + drafter patch + weight sha256 check
scripts/bench/                 named-test concurrency sweep, C1/natural probes,
                               prompt processing + TTFT, realistic suite, surface
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

Every cell is a measurement taken on this host with the configuration above, and each
one is the **median of 3 reps** — a single number, deliberately, so two cells can be
compared without deciding which of several recordings to believe. `aggregate_tok_s` is
the metric for the token-generation rows.

A **test name** pins the prompt *and* the `Max Tokens` budget together, from one table
in `scripts/bench/conc.py`: the prompt and the budget are what the number measures, so
they travel with the name and cannot be recombined. Each budget is the prompt's natural
completion length, measured by running it to a stop rather than guessed; Coding is the
one that never stops inside its budget, so its figure is a cap.

| Test Name | Concurrency | Tokens Generated | Value | Command |
|---|---:|---:|---:|---|
| Technical Planning | 10 | 2048 | **239.70** | `conc.py --test planning --cells 10:30` |
| Technical Planning | 5 | 2048 | 161.27 | `conc.py --test planning --cells 5:20` |
| Technical Planning | 1 | 2048 | 45.81 | `conc.py --test planning --cells 1:6` |
| Counting | 10 | 512 | **675.59** | `conc.py --test counting --cells 10:30` |
| Counting | 5 | 512 | 473.83 | `conc.py --test counting --cells 5:20` |
| Counting | 1 | 512 | 136.28 | `conc.py --test counting --cells 1:6` |
| Prose | 10 | 2048 | **210.50** | `conc.py --test prose --cells 10:30` |
| Prose | 5 | 2048 | 135.16 | `conc.py --test prose --cells 5:20` |
| Prose | 1 | 2048 | 38.26 | `conc.py --test prose --cells 1:6` |
| Coding | 10 | 2048 | **325.96** | `conc.py --test coding --cells 10:30` |
| Coding | 5 | 2048 | 224.31 | `conc.py --test coding --cells 5:20` |
| Coding | 1 | 2048 | 61.37 | `conc.py --test coding --cells 1:6` |
| Prompt Processing | 1 | — | **2 819** tok/s | `prefill.py --length-sweep` |
| TTFT | 1 | — | **210.7** ms | `prefill.py --test …` (median over the four test prompts) |

Concurrency is the number of requests in flight, capped by `MAX_NUM_SEQS`; the top cell
of each test is the state a busy engine is actually in. `—` on the last two rows means
the budget is not part of that metric: both are read from the first token of a stream,
which does not depend on how many tokens are asked for afterwards.

`--max-tokens` is part of the metric, not a knob: `aggregate_tok_s` moves ~15% between
96 and 256 at `conc=10`, which is what once made a correct engine look ~13% depressed.
What the spread actually is, how each number above was produced, why Prompt Processing
has to be measured as a slope and not a ratio, which third-party numbers on this
hardware are **not** parity targets, and which figures from the investigation were
eager-era and are not comparable to anything above:
[docs/measurement-notes.md](docs/measurement-notes.md). The checks that do not fit this
table — acceptance length, correctness, serving surface, graph capture, soak, and the
single-stream and realistic-suite probes — are in
[docs/recorded-metrics.md](docs/recorded-metrics.md).

## Qwen3.8-27B-INT4 Recorded Performance Metrics

> **PLACEHOLDER — this repo holds no INT4 measurement.**
>
> Reserved for the INT4 profile's numbers, in the same shape and with the same metric
> names as the FP8 table above. The `how` column pins every parameter that moves the
> number, mirroring the FP8 canonical set so the two profiles stay comparable —
> measure at those values, or change the column and the cell together:
>
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
>
> The supporting checks that sit outside this grid — acceptance length, correctness,
> serving surface, graph capture, soak, and the single-stream and realistic-suite
> probes — take their column shape from
> [docs/recorded-metrics.md](docs/recorded-metrics.md), which is where the FP8
> equivalents are recorded. Fill them from `scripts/verify.sh` and
> `scripts/bench/run-all.sh` on an idle engine, three reps per cell, median published,
> and only from the harnesses in this repo.
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
| [docs/recorded-metrics.md](docs/recorded-metrics.md) | every measurement the README does not show: acceptance, correctness, surface, graph capture, soak, the single-stream probes, the realistic suite |
| [docs/measurement-notes.md](docs/measurement-notes.md) | how to read the metrics table: where each recording comes from, why the prefill rate is a slope, the `max_tokens` mismatch, spread, eras, and the third-party numbers that are not parity targets |
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
