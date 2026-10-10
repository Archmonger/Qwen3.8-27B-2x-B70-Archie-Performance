# Recorded metrics: everything the README does not show

[The README](../README.md#qwen38-27b-fp8-recorded-performance-metrics) carries the
summary grid — the four token-generation tests at three concurrencies, plus prompt
processing and TTFT. This file holds the rest of what has been measured on this
configuration: the supporting checks, the single-stream probes, and the engine-level
attestations. Every value here was produced by a harness in this repo on this host, in
the same session as the README's grid, against a container that `verify.sh` had just
attested.

How to read a number here: quote the **metric name** with it. `aggregate_tok_s`
(`conc.py`), `median_tok_s_after_ttft` (`probe.py`), `tok_s_1_100_after_ttft`
(`suite.py`), and the engine log's "Avg generation throughput" are four different
measurements and are not interchangeable. [measurement-notes.md](measurement-notes.md)
covers why, and what each protocol is worth.

## Supporting checks

| test name | metric | recorded | how |
|---|---|---|---|
| Acceptance length (C1) | mean accepted tokens per draft step | **7.500** (26 drafts / 169 accepted) | `acceptance.sh` |
| Acceptance length (natural) | mean accepted tokens per draft step | **3.000** (256 / 512) | `acceptance.sh` |
| Correctness | correct answers / attempted | **15/15** this session; **30/30** per graphs-on arm of the establishing A/B | `correctness.sh` |
| Serving surface | checks passed | **5/5** | `surface.py` |
| Graph capture cost | GiB/rank, seconds | **0.58-0.60** GiB/rank, **7** s | engine log |
| Configured capture ladder | captured batch sizes | `[1,2,4,8,16,24,32,40,48,56,64,72,80]` (13 sizes to 80) | engine log (`cudagraph_capture_sizes`) |
| Soak | cycles, duration, load rounds, wedges, restarts | **6/6** cycles, **600** s each (**3600** s), **46** completed load rounds, **0** wedges, **0** restarts | `soak.sh --profile fp8` |

The acceptance pair is the strongest single signal that a build is the build it claims
to be: landing on the *same* draft and accepted counts (`C1 26 169 7.500`,
`NAT 256 512 3.000`) is a fingerprint of drafter, target, and `k` taken together, and
it held across every recording. At `k=7` the ceiling is 8.000; a zero-draft reading
means spec decode disabled itself, which is what `acceptance.sh` exists to catch.

The correctness row is not decoration. The graph-stride defect in
[cudagraph-stride-defect.md](cudagraph-stride-defect.md) runs at full throughput with
healthy acceptance counters while producing corrupt output, so a throughput table on
its own can be entirely green and entirely wrong.

## Single-stream probes (`probe.py`)

These predate the per-test grid and measure something the grid does not: one request,
no queue, counted from SSE token events rather than from server-reported usage.

| test name | prompt | budget | `median_tok_s_after_ttft` | how |
|---|---|---|---|---|
| Short prompt (C1) | `"1 2"` (15 tok reported) | 16 | **151.68** (reps 150.26 / 151.81 / 151.68, peak 152.83) | `probe.py --prompt "1 2" --max-tokens 16 --iters 10 --warmup 3 --label C1` |
| Natural (planning prompt) | `prompt-planning.txt` | 96 | **58.98** (reps 58.98 / 59.00 / 58.63, peak 59.18) | `probe.py --prompt "$(cat data/prompt-planning.txt)" --max-tokens 96 --iters 6 --warmup 2 --label NAT` |

Each row is the median of three `probe.py` runs taken in the same session as the
README's grid, on the attested boot. The per-rep values are printed rather than a bare
range so the spread is inspectable: C1 varies ~1.0% rep to rep, natural ~0.6%. Both
hold the boot fixed, so they measure rep noise only — not a session-to-session floor.
See the spread section of [measurement-notes.md](measurement-notes.md).

The C1 row's prompt is the same 15-token baseline `prefill.py` pairs against, which is
what makes the two readable together: `probe.py` counts SSE token events on one stream
while `prefill.py` reads the server's own usage for the same prompt. These are the
numbers to compare against C1-style published figures, and they are *not* comparable to
the README's grid: the grid is a server-reported aggregate under concurrency, these are
single-stream and SSE-counted.

The natural row is quoted at `--max-tokens 96`, which is its canonical budget for the
same reason the grid rows are pinned — the number moves with the budget, and the ~59
figure has only ever been measured at 96.

## Realistic suite (`suite.py`)

| metric | recorded | how |
|---|---|---|
| `tok_s_1_100_after_ttft` | **42.11** (runs 42.11 / 42.55 / 42.03, each n=12) | `suite.py`, 12 cold prompts from `data/realistic-suite-v1.json`, each sent once |
| Suite TTFT | **165.6** ms (runs 165.6 / 140.6 / 179.6) | same run |

The suite's per-run median is stable (42.03-42.55, a 1.2% spread) while its TTFT is not
(140.6-179.6 ms). That is not a defect in the number, it is what the two metrics are
made of: each run sends 12 *distinct* prompts once, so the throughput is averaged over
12 samples of the same 12 prompts, whereas TTFT is dominated by whether a request
happens to arrive in the fast or slow mode of the session (see the drift discussion in
[measurement-notes.md](measurement-notes.md)). Quote the throughput; treat a single
suite TTFT as a single sample.

The suite is the closest thing here to a real workload: 12 distinct cold prompts, one
request each, so nothing is cached and nothing repeats. That is also why its TTFT sits
apart from the README's TTFT row — different prompts, different prompt lengths, and
12 samples of a queue-free single stream rather than the per-test prompts.

## Why the README's grid does not include these

Three reasons, all of them about the table being readable rather than about these
measurements being weak:

- **Different metric, different counting rule.** Mixing `median_tok_s_after_ttft`
  (SSE events, one stream) with `aggregate_tok_s` (server usage, N concurrent threads)
  in one grid invites exactly the comparison that has already produced a false parity
  conclusion on this hardware.
- **No concurrency axis.** The grid's whole point is that throughput is read across
  C10 / C5 / C1. A measurement that only exists at C1 does not belong in that axis.
- **One is per-workload, not per-test.** The suite's TTFT is a property of its 12
  prompts; the README's TTFT row is per test prompt, measured by `prefill.py`.
