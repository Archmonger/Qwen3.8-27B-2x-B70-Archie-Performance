# Verification discipline (and the harness bugs that taught it)

The rule this repo operates on: **acceptance is functional, not status.** An
engine is "working" when it produces the right artifact — correct text, drafts
flowing, a tool call parsed — not when it starts, not when `/health` returns 200,
and not when a config echoes the number you set. This image has no `HEALTHCHECK`
and runs with `network_mode: host`, so `/health` is a liveness signal only.

Several of the traps below are recorded because a broken check cost days, and a
check that cannot fail is worse than no check: it produces confidence.

## What `scripts/verify.sh` proves, and why each check exists

| check | what a pass means | what it catches |
|---|---|---|
| `/health` + container running | something is listening | a crash loop |
| fatal-signature grep of the log | no `Traceback`/`DEVICE_LOST`/`OUT_OF_RESOURCES` | the driver-overlay failure that otherwise surfaces as a "slow boot" |
| engine-printed config | the **engine** has the ceiling / mode / k / method | an env var vLLM silently ignored, or an image rebuild that dropped a flag |
| capture **ladder** contains `MAX_NUM_SEQS x (1+k)` | the saturated batch is actually graphed | the 72-vs-80 defect: echoing `max_cudagraph_capture_size: 80` is not proof the ladder contains 80 |
| `Graph capturing finished ... took N GiB` present | graphs were captured at all | graphs silently off, everything eager |
| V2 model runner confirmed | dspark is actually wired | V1 fallback, where dspark does not exist |
| 15 correctness probes | the text is right | **the stride defect — invisible to every other check** |
| acceptance windows | drafts are flowing and being accepted | spec decode silently disabled |
| 5 serving-surface cells | thinking, tool calls, reasoning, usage details, code | a flag combination that breaks the chat template |

## Prove your tests can fail

`scripts/falsify/run-all.sh` drives every guard with inputs that must make it fail:
a synthetic engine log with each attested field corrupted one at a time, a mutated
profile for each `setup.sh` preflight guard, and scripted `/metrics` counters for the
two spec-decode guards. Each harness also runs a **control** — the unmutated input
must pass — because a case that only "discriminates" against an already-broken
baseline proves nothing.

Two real failures this class caught in this project:

- a summary that printed `observed CORRUPT` from **zero** parsed rows (vacuous
  pass — an aggregate over empty input looks like a verdict);
- a boot counter that counted **lines** (17/boot) instead of boots.

And one caught while building this repo: the falsifier's own fixture was written in
a prettified config format, which silently failed every extraction regex — so the
attestation would have reported "all rules bad" on a healthy engine and nobody would
have noticed the regexes matched nothing in production. The fixture is now
transcribed from a real engine log, and `CONTAINER=<name> scripts/falsify/run-all.sh`
cross-checks it against one.

Related shapes to distrust: an unreachable pass condition, an aggregate hash over
empty input, a `grep -qF && good++ || [[ ... ]] && echo WRONG` that prints `WRONG`
on **success** (it parses as `((A&&B)||C)&&D` — use explicit if/else),
`awk 'BEGIN{exit !(x<1.5)}' || fire`, which fires on a healthy value and lets a
broken one through, and `cmd | grep -q .` under `pipefail`, where grep's early exit
SIGPIPEs `cmd` and the pipeline reports failure for content that exists.
`acceptance.sh` tests the predicate positively and says why.

## Harness bugs that misreported a *healthy* engine

All three are fixed with **guards**, not just patches, so the class of bug stays closed:

1. **Correctness reported 12/18 on a good engine.** The prompts were piped through
   `while IFS= read -r`, which split the one multi-line prompt
   (`def add(a, b):\n    return`) into two prompts — so it tested something the
   measured arms never tested. Now: a bash array, plus a hard gate that the run
   executed exactly `${#prompts[@]} * REPS` probes. A run of 12/18 is a harness
   bug and exits non-zero rather than printing a rate.
2. **A soak recorded a healthy engine as TIMEOUT — and, once fixed, nearly recorded a
   wedge as PASS.** `rc=143` was tallied as a stall after the script's own kill. Now:
   `rc=143` is INCONCLUSIVE, and a run with any inconclusive cycle exits 2 rather than
   reporting PASS. (`timeout --preserve-status` is deliberately *not* used — it would
   report 143 on a genuine timeout and destroy the 124-vs-143 distinction.)
   Two further defects in the same script were found while re-recording the soak row:
   it referenced an unset `$MNAME` under `set -u` and aborted cycle 1 (a clean exit, so
   a run that applied *no load at all* read as an engine problem); and it applied one
   burst per cycle, which used to fill the window but drains in ~75 s now that CUDA
   graphs doubled decode — so most cycles came back INCONCLUSIVE, i.e. the soak spent
   87% of its time idle while claiming to soak.
   The fix repeats the load until the deadline, which exposed a *third* bug worth
   naming: reaching the deadline is not evidence of a healthy engine. An engine wedged
   inside oneCCL also reaches 600 s, keeps answering `/health`, and completes nothing —
   so the naive pass condition was satisfiable by the very failure the soak exists to
   catch. A cycle now requires **both** the elapsed window **and** at least one load
   round that completed under its own power; zero completed rounds is a FAIL, not a
   pass. `--load-cells`, `--cycles` and `--cycle-seconds` are overridable so the guard
   can actually be exercised rather than assumed.
3. **A `--max-tokens` default silently changed the metric, and the engine was blamed.**
   `conc.py` computed `aggregate_tok_s` as counted `completion_tokens / wall`, but
   wall time also pays for prefill. Its default was 96 while every recorded number —
   and both scripted callers — used 256, so a hand-run sweep compared 96 against a
   256 record and came out ~13% low. Because the fixed prompt is a few hundred tokens
   (server-reported `prompt_tokens`: 286 counting, 296 planning and prose, 347 coding),
   the uncounted prefill cost amortises over `max_tokens`, so the loss appeared at
   `conc=10` and barely at `conc=1` — which is exactly the signature of a "host-side
   condition". (This section used to cite a "~243-token" prompt and attribute the
   single-stream immunity to prefix caching. Both were wrong: 243 is the `wc -w` *word*
   count of the planning file, and prefix caching is not engaged at these sizes at all,
   so it cannot be making prefill free. The measured ~13%/~4% asymmetry stands; the
   mechanism does not. See [measurement-notes.md](measurement-notes.md).) Verified on one boot
   with alternating arm order, changing only the flag: conc=10 **193.23-202.94** at
   96 vs **229.16-231.64** at 256, against a recorded 227.13. An out-of-sample point
   at 160 interpolated to ~211 and measured 214.82.
   Now: `CANONICAL_MAX_TOKENS = 256` is the default, `max_tokens` is printed with
   every cell and in `--json`, and a non-canonical value exits 3 unless
   `--confirm-noncanonical` is passed. Proven able to fail both ways by
   `scripts/falsify/conc-cases.sh`, including a mutation check that the harness goes
   red if the guard is deleted. The same class of error also inflated the natural
   probe gap (50.21 quoted at `--max-tokens 512` against a ~59 record taken at 96).

## Measurement rules

- **Measure at the `MAX_NUM_SEQS` cap.** `MAX_NUM_SEQS=10` means a saturated engine
  is always at the 10-request cap; measuring at conc=8 hid a 38% batched loss
  across two deploys.
- **Pair your A/Bs and alternate arm order.** ~5% drift per 10 boots on the natural
  probe. Two claims died once order was controlled.
- **Pin and echo every harness parameter that moves the number.** `--max-tokens`
  changes `aggregate_tok_s` by ~15% (rule 3 above), so a comparison where the two
  sides used different values is not a comparison. This is a third category of
  error, distinct from noise and from drift: the measurement was *precise*, it
  measured a different metric. Before attributing a gap to spread, confirm both
  sides used the same `max_tokens`, the same cells, and the same probe.
- **Measure the noise floor instead of picking a threshold.** Boot the identical
  config as a calibration arm and use its spread. A hardcoded 3% once promoted a
  +3.4% artifact to "falsified" on a probe whose real spread was much larger.
  Compute the floor **per probe**: pooling the counting probe with the natural
  prompt gave a 90% floor that muted a real +38%. Note that any historical spread
  figure predating the `max_tokens` finding is suspect, since a mismatched
  `max_tokens` produces a large, uniform, perfectly reproducible pseudo-spread.
  Reps **on one boot** with every harness canonical spanned ~1% (C1 151.12-151.98,
  natural 58.69-58.92, conc=10 at 256 tokens 229.16-231.64); that is a within-boot
  figure and is deliberately not offered as a session-to-session floor, because it
  holds boot, box and cache state fixed while this host drifts ~5% per 10 boots.
- **State the metric with every number.** `tok_s_1_100_after_ttft` from
  `scripts/bench/suite.py`, `aggregate_tok_s` from `conc.py`, and
  `median_tok_s_after_ttft` from `probe.py` are three different things. Engine-log
  "Avg generation throughput" is an aggregate and comparable to none of them.
- **Count tokens the same way the references do.** Ask for `return_token_ids` so one
  SSE event == one generated token; counting raw SSE *text* events understates a
  spec-decoding server catastrophically (7.6 vs 38.1 tok/s on the same stream).
- **Reject a spec arm with zero drafts while the engine is alive**, and gate liveness
  immediately before each probe. A dead engine recording `0 0 n/a` looks like a
  valid measurement of spec decode with no drafts.

## Reproducing a number from the README

```bash
sudo ./setup.sh --profile fp8                 # attests + verifies on the way up
./scripts/bench/run-all.sh --profile fp8      # idle engine, prints all three metrics
./scripts/soak.sh --cycles 6 --cycle-seconds 600
```

If a check you added cannot be made to fail, it is not a check.
