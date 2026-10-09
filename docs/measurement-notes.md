# How to read the recorded metrics

The table in
[the README](../README.md#qwen38-27b-fp8-recorded-performance-metrics) is the
summary; this document is the reasoning that makes each row quotable, and the record
of the claims this investigation already got wrong so they are not re-derived as new
findings.

## Where each recording comes from

The single `recorded` column mixes protocols on purpose — every value is a measurement
of this configuration on this host — but the reps are not interchangeable, so do not
pool them into one mean or treat a range as a confidence interval:

- **The establishing paired A/B** (2 reps per arm, alternating boot order) produced the
  first value in each `conc.py` row: conc=10 **227.13**, conc=12 **221.50**, conc=9
  212.42, conc=8 202.56, conc=1 42.6, and the 30/30 correctness-per-arm rows. This is
  the arm that decided the configuration, and it is the number to compare a change
  against. The soak row came from the same investigation phase rather than from an A/B
  arm.
- **Harness re-runs against the live container** produced the remaining batched values
  (229.59, 230.47 at conc=10; 228.81, 230.41 at conc=12) and the ranges at conc=9,
  conc=8, and conc=1. These were taken after the container was stood up from this
  repo's own files, so their job is to show the configuration produces the same
  numbers from a clean build — they are single or few runs, not additional A/B arms.
- **Within-boot reps** produced the C1 **151.12-151.98** (peak 152.65), natural
  **58.69-58.92** (peak 58.94), and suite **40.21-41.58** (3 runs) rows, plus the
  229.16-231.64 conc=10 range quoted in the `--max-tokens` section below. These hold
  the boot, box state, and cache state fixed, so they measure rep noise only — see
  the spread section.

The establishing A/B logged the two `probe.py` rows only approximately (~152 C1, ~59
natural), each within 0.5% of the rep peaks (152.65, 58.94), so those rows carry the
precise within-boot reps rather than a merged cell that would imply the approximations
were measurements of the same kind.

The acceptance windows landing on the **same** draft and accepted counts across every
recording (`C1 26 169 7.500`, `NAT 256 512 3.000`) is the strongest signal that a
clean build boots the same engine: those counters are a fingerprint of the drafter,
the target, and k together.

Use `../scripts/bench/run-all.sh` to take the whole table again in one go.

## `--max-tokens` is part of the metric, not a knob

`conc.py`'s `aggregate_tok_s` divides counted `completion_tokens` by wall time, but
wall time also pays for prefill. The prompt here is a fixed ~243 tokens, so that
uncounted prefill cost is amortised over `max_tokens`: at `conc=10`, 96
tokens/request reads ~13% below 256 on the *same boot*, while `conc=1` moves only
~4% because prefix caching makes prefill nearly free for a single stream. That
asymmetry is what made it look like a host problem.

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
parameter, repeated reps **on one boot** gave C1 **151.12-151.98** (peak 152.65), the
natural 512-token probe **58.69-58.92**, and conc=10 at `--max-tokens 256`
**229.16-231.64** — a ~1% range.

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

The realistic-suite metric for this configuration is **~40-42 tok/s**. Do not compare
that against the 15.67 that appears in the investigation's working notes:
15.67 was measured in the **eager, k=4** era, before graphs were enabled at all, and
it is not comparable to anything in the README table. Likewise the "~1.6x from spec
decode" figure (13.76 DSpark k=4 vs 8.60 no-spec on `vllm bench serve`) and the
spec-off suite baseline of 10.32 are eager-era, k=4 measurements — useful as a
same-era A/B, useless as context for the graphed config.

Always state which metric you are quoting. `tok_s_1_100_after_ttft` (`suite.py`),
`aggregate_tok_s` (`conc.py`), `median_tok_s_after_ttft` (`probe.py`), and the
engine log's "Avg generation throughput" are four different measurements.

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
