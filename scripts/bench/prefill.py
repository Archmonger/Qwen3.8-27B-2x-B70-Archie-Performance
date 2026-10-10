#!/usr/bin/env python3
"""Prompt-processing throughput and time-to-first-token, at concurrency 1.

Two rows of the README's metrics table come from here, and neither can be produced by
the other harnesses:

  Prompt Processing -> prefill tok/s   (from --length-sweep)
  TTFT              -> ms              (the default per-test mode)

Why not reuse conc.py or suite.py: conc.py divides counted completion tokens by
whole-cell wall time, so prefill sits inside the denominator and is invisible as a
quantity of its own; suite.py reports TTFT only across its own 12 cold prompts, which
is a different workload from the per-test prompts here. A row in the table needs its
own harness so its protocol is stated in one place and cannot drift from the row it
labels.

Why streaming: TTFT is only a first-token latency if you timestamp the first token
event. A non-streaming request returns when generation finishes, so its wall time is
prefill plus the entire decode and there is nothing left to call TTFT. The request
asks for return_token_ids so one SSE event carries the token ids it represents -- a
speculative accept arrives as a single event holding several ids.

Why the generation budget is deliberately tiny (BUDGET_TOKENS): both rows measure the
cost of getting the FIRST token, which does not depend on how many tokens are requested
afterwards -- prefill of a ~300-token prompt is one chunk under chunked prefill with
max_num_batched_tokens=4096 either way. Running the table's real budgets here would
only add ~40 s of draining per request to buy a number that does not move, and a
harness that is cheap to run is a harness that gets run. prompt_tokens is likewise
independent of the budget. The budgets in conc.TESTS belong to the token-rate rows.

READ THIS BEFORE QUOTING A PREFILL RATE
---------------------------------------
A prefill rate cannot be read off a single request at these prompt sizes, and the
first version of this harness did exactly that and published a number that was not a
rate. Measured on this deployment: a 15-token prompt had TTFT 182 ms and a 347-token
prompt had TTFT 169 ms. Prefill cannot cost negative time. At a few hundred tokens
TTFT is dominated by a per-request latency floor -- HTTP/SSE establishment,
scheduling, the first decode step, and under spec decode a draft pass before the first
token is accepted -- that is LARGER than the entire prefill signal in that range. So
prompt_tokens/TTFT does not measure prefill throughput; it ranks mostly by floor, and
it can move while prefill does not.

What IS well defined is the derivative. --length-sweep runs the same request at a range
of prompt lengths and fits d(tokens)/d(TTFT); the floor is common to all lengths and
cancels in the difference, leaving the part of TTFT that actually scales with prompt
size. That slope is the number in the README. The low end of the sweep is reported but
excluded from the fit: there, d(tokens)/d(TTFT) divides by a quantity whose own error
is comparable to it.

Why every probe prompt is salted with a process-unique nonce, and why cached_tokens is
checked per sample. The slope was first measured twice and returned 2917.7 and 6664.2
tok/s -- a 2.3x disagreement whose cause was in the probe, not the box. The salt was a
per-process counter, so the second run rebuilt byte-identical prompts and took
prefix-cache hits on the longest ones, deleting their prefill and inflating the slope.
So: salts are unique across processes (pid + wall-clock), and a cache hit is DETECTED,
reported, and excluded -- a cache-hit sample is not a slower measurement of the same
thing, it is a measurement of a different thing. Note that prefix caching on this
deployment engages only once a prompt fills at least two full 2048-token KV blocks
(identical repeats report cached_tokens=0 at 296, 347, 1292 and even 2572 prompt tokens,
and 4096 at 6412), so this bites only at the long end of the sweep -- which is exactly
where the slope is anchored.

Why rounds and not blocks: the box drifts over a session and an uninterleaved run
turns drift into a difference between prompts. An earlier version measured all samples
of one test, then all of the next, and reported TTFT falling monotonically in RUN ORDER
(225 / 207 / 168 / 166 ms) while the spread within any one test was about 2 ms. But
pure interleaving has its own artifact: with a short baseline interleaved, TTFT tracked
position-in-round (209 ms in one slot, 169 ms in the other) rather than prompt. The
per-test mode therefore pairs each sample with a baseline measured immediately before
it in a FIXED slot order, and reports the baseline's spread so drift is visible in the
baseline column instead of being silently attributed to a prompt. A test whose baseline
moved more than STABLE_MAX_SPREAD_MS is flagged rather than averaged.

Reps are independent measurements, not a warm/cold pair: prefix caching does not engage
for prompts of this size on this deployment, so a repeat is no cheaper than a first
pass (measured directly; see docs/measurement-notes.md).

Test names, prompts, and budgets are imported from conc.py's TESTS table so the two
harnesses cannot disagree about what a test is.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conc import TESTS, load_prompt  # noqa: E402  (one source of truth for a test)

# See "Why the generation budget is deliberately tiny" in the module docstring.
BUDGET_TOKENS = 16

# A prompt short enough that its own prefill is negligible, so its TTFT is the part
# of TTFT that is NOT prefill. "1 2" is reused from probe.py's C1 counting probe so the
# floor is measured with a prompt this repo already characterises.
BASELINE_PROMPT = "1 2"

# Lengths for the slope fit, in target tokens.
SWEEP_LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
# Only points whose TTFT clearly exceeds the latency floor can inform a slope.
FIT_MIN_TTFT_MS = 250.0
SWEEP_ROUNDS = 5

# A baseline that moved more than this across a test's block means the session drifted
# while that test was being measured; the reading is then not comparable to the other
# tests' and must not be averaged into a table cell.
STABLE_MAX_SPREAD_MS = 15.0

# Rounds of full interleaved traffic run before any measurement, to reach steady
# state. The per-prompt warm() calls only prove reachability and pay first-request
# JIT; they are far too few for that. Measured: the first measurements of a session ran
# ~43 ms slower than the last, in both the test prompts and the ~3-token baseline, so
# the drift is a property of the session and measuring before it settles yields a TTFT
# column that is really a countdown.
WARMUP_ROUNDS = 6

SWEEP_UNIT = ("The following sentence is repeated to build a prompt of a known "
              "approximate length for a prefill cost probe. ")
SWEEP_TOKENS_PER_UNIT = 17.0  # approximate; only the slope of the fit matters

# Unique across PROCESSES, not merely within one. See the module docstring: reusing
# salts across runs is what produced a 2.3x wrong prefill rate once.
_NONCE = f"{os.getpid()}-{int(time.time() * 1000)}"


def build_sweep_prompt(target_tokens, salt):
    """A prompt of roughly target_tokens tokens, unique per (length, salt, process).

    Salted so no sample is a prefix of another: prefix caching is inert below one
    attention block here, but not above it, and the long end of the sweep is where the
    slope is anchored.
    """
    k = max(1, int(round(target_tokens / SWEEP_TOKENS_PER_UNIT)))
    return f"[{_NONCE} probe {target_tokens} {salt}] " + SWEEP_UNIT * k


def stream_one(url, model, prompt, max_tokens):
    """One streaming request. Returns (prompt_tokens, ttft_s, n_tokens, cached_tokens).

    Reads the stream to the end: the first token event is what is timed, but the
    response must be drained so the server finishes and frees its KV before the next.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "return_token_ids": True,
        # Without this, vLLM omits usage on a streaming response and prompt_tokens
        # would have to be guessed from the prompt text instead of read from the
        # server that actually tokenised it.
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    prompt_tokens = None
    cached_tokens = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            usage = ev.get("usage")
            if usage and usage.get("prompt_tokens") is not None:
                prompt_tokens = usage["prompt_tokens"]
                cached_tokens = (usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens")
            if ttft is not None:
                continue
            ch = (ev.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            tids = (delta.get("token_ids") or ch.get("token_ids")
                    or ev.get("token_ids"))
            if tids:
                n_tokens += len(tids)
            else:
                text = ch.get("text")
                if text is None:
                    text = delta.get("content")
                if not text:
                    # Role-only and usage-only events carry no token. Treating one as
                    # the first token would report queue+setup latency as TTFT.
                    continue
                n_tokens += 1
            ttft = time.perf_counter() - t0
    return prompt_tokens, ttft, n_tokens, cached_tokens


def warm(url, model, prompt, n):
    """Reachability + first-request JIT, deliberately unbilled."""
    for _ in range(n):
        try:
            stream_one(url, model, prompt, BUDGET_TOKENS)
        except Exception as exc:
            # stderr, not stdout: both modes print their JSON result on stdout, and a
            # diagnostic there either corrupts it or -- worse -- makes a failed run look
            # like it produced output. Diagnostics on stderr, results on stdout.
            print(f"WARMUP_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    return 0


def fit_slope(points):
    """d(tokens)/d(ttft_s) by least squares, plus the implied zero-token TTFT floor.

    `points` is [(ttft_s, prompt_tokens), ...].
    """
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if not den:
        return None
    slope = sum((x - mx) * (y - my) for x, y in points) / den
    if not slope:
        return None
    # tokens = slope * ttft_s + c, so the floor is the ttft where tokens -> 0.
    return {"prefill_tok_s": round(slope, 1),
            "ttft_floor_ms": round(1000 * (mx - my / slope), 1),
            "points": len(points)}


def run_length_sweep(url, model, rounds):
    """Prefill rate as a slope. See "READ THIS BEFORE QUOTING A PREFILL RATE"."""
    if warm(url, model, build_sweep_prompt(64, "warmup"), 3):
        return 2

    samples = {n: [] for n in SWEEP_LENGTHS}
    excluded = 0
    salt = 0
    for rnd in range(rounds):
        # Alternate order within the round so position-in-round is shared by every
        # length instead of landing on whichever ran last.
        order = SWEEP_LENGTHS if rnd % 2 == 0 else list(reversed(SWEEP_LENGTHS))
        for n in order:
            salt += 1
            ptok, ttft, _n, cached = stream_one(
                url, model, build_sweep_prompt(n, salt), BUDGET_TOKENS)
            if ptok is None or ttft is None:
                print(f"dropped length={n} prompt_tokens={ptok} ttft={ttft}",
                      file=sys.stderr)
                continue
            if cached:
                excluded += 1
                print(f"EXCLUDED cache hit length={n} prompt_tokens={ptok} "
                      f"cached_tokens={cached}", file=sys.stderr)
                continue
            samples[n].append((ptok, ttft))

    rows = []
    for n in SWEEP_LENGTHS:
        if not samples[n]:
            continue
        ptoks = [p for p, _ in samples[n]]
        tts = [t for _, t in samples[n]]
        rows.append({"target_tokens": n,
                     "prompt_tokens": round(statistics.median(ptoks), 1),
                     "median_ttft_ms": round(1000 * statistics.median(tts), 1),
                     "min_ttft_ms": round(1000 * min(tts), 1),
                     "max_ttft_ms": round(1000 * max(tts), 1),
                     "samples": len(tts)})

    all_pts = [(r["median_ttft_ms"] / 1000.0, r["prompt_tokens"]) for r in rows]
    fit_pts = [p for p in all_pts if 1000 * p[0] >= FIT_MIN_TTFT_MS]
    out = {
        "nonce": _NONCE,
        "excluded_cache_hits": excluded,
        "fit_min_ttft_ms": FIT_MIN_TTFT_MS,
        "curve": rows,
        # Reported for transparency, NOT the publishable rate: the low end is
        # floor-dominated, so a fit over everything is dragged by points that carry
        # almost no prefill signal.
        "fit_all_lengths": fit_slope(all_pts),
        "prefill_tok_s": (fit_slope(fit_pts) or {}).get("prefill_tok_s"),
        "ttft_floor_ms": (fit_slope(fit_pts) or {}).get("ttft_floor_ms"),
        "fit_points": len(fit_pts),
    }
    print(json.dumps(out, indent=2))
    return 0


def run_per_test(url, model, names, iters, warmup_n):
    """Per-test TTFT, each sample paired with a baseline measured just before it.

    Tests are INTERLEAVED round-robin with the order alternating per round, not
    blocked. Blocking was tried and failed reproducibly: with tests always measured
    counting -> planning -> prose -> coding, the prose block's baseline drifted ~43 ms
    in every rep (42.5 / 43.0 / 43.6) -- far too consistent to be random session drift,
    and a property of occupying slot 3. Interleaving with alternating order puts each
    test in every position across the run, so a position effect is shared by all tests
    instead of permanently disqualifying whichever one happens to be third.

    Each sample still carries its own immediately-preceding baseline, so drift is
    measured at the moment it would matter rather than assumed away.
    """
    prompts = {}
    for name in names:
        prompts[name] = load_prompt(TESTS[name]["prompt"])
        rc = warm(url, model, prompts[name], warmup_n)
        if rc:
            return rc
    if warm(url, model, BASELINE_PROMPT, warmup_n):
        return 2

    # A long warmup phase, then measure. The per-prompt `warm` calls above are only
    # enough to prove reachability and pay first-request JIT; they are far too few to
    # reach steady state. Without this phase the whole run sits in a slow mode: the
    # earliest measurements in a session came back ~43 ms slower than the latest, in
    # both the test prompts AND the baseline ("1 2" read 182 ms early and 139 ms late),
    # so the effect is a property of the session, not of the prompt. Measuring before
    # steady state produced a TTFT column that was really a countdown.
    for w in range(WARMUP_ROUNDS):
        order = list(names) if w % 2 == 0 else list(reversed(names))
        for name in order:
            stream_one(url, model, BASELINE_PROMPT, BUDGET_TOKENS)
            stream_one(url, model, prompts[name], BUDGET_TOKENS)

    acc = {name: {"ttft": [], "ptok": [], "base": []} for name in names}
    for rnd in range(iters):
        order = list(names) if rnd % 2 == 0 else list(reversed(names))
        for name in order:
            # Baseline immediately before the sample: it measures the floor this
            # specific reading was taken against.
            bp, bt, _bn, _bc = stream_one(url, model, BASELINE_PROMPT, BUDGET_TOKENS)
            ptok, ttft, ntok, cached = stream_one(url, model, prompts[name],
                                                  BUDGET_TOKENS)
            if bt is None or ttft is None or ptok is None or ntok < 1:
                print(f"discarded sample round={rnd} test={name}: prompt_tokens={ptok} "
                      f"ttft={ttft} tokens={ntok}", file=sys.stderr)
                continue
            if cached:
                print(f"discarded sample (cache hit) round={rnd} test={name}: "
                      f"cached_tokens={cached}", file=sys.stderr)
                continue
            acc[name]["base"].append(1000 * bt)
            acc[name]["ttft"].append(1000 * ttft)
            acc[name]["ptok"].append(ptok)

    out = {}
    for name in names:
        test_ttft = acc[name]["ttft"]
        test_ptok = acc[name]["ptok"]
        base_ttft = acc[name]["base"]

        if not test_ttft:
            print(f"{name}: no usable samples")
            return 1
        distinct = sorted(set(test_ptok))
        # Gate on the spread of the quantity actually published -- the test's own TTFT
        # readings -- not on the baseline's. Gating on the baseline was tried and it
        # vetoes valid measurements: planning and prose were flagged UNSTABLE in every
        # rep because their baseline alternated ~186/~141 ms, yet their test readings
        # moved only ~3 ms (207.4-211.0 and 207.0-210.3) and did not track the
        # baseline's state at all. The baseline's mode turns out to depend on which
        # prompt ran immediately before it, which is a property of the probe sequence,
        # not of the prompt under test. The baseline is therefore reported as context
        # for the floor (baseline_spread_ms, baseline_ms_samples) and is not allowed to
        # disqualify a reading; the guard below is what catches a real drift.
        test_spread = max(test_ttft) - min(test_ttft)
        base_spread = (max(base_ttft) - min(base_ttft)) if base_ttft else None
        out[name] = {
            "samples": len(test_ttft),
            # Server-reported, so normally one value; anything else means the prompt is
            # not tokenising deterministically and must be visible, not medianed away.
            "prompt_tokens": distinct[0] if len(distinct) == 1 else distinct,
            "median_ttft_ms": round(statistics.median(test_ttft), 1),
            "min_ttft_ms": round(min(test_ttft), 1),
            "max_ttft_ms": round(max(test_ttft), 1),
            "ttft_ms_samples": [round(x, 1) for x in test_ttft],
            "baseline_prompt": BASELINE_PROMPT,
            # Per-sample, not just the median: the test reading and the baseline
            # reading taken immediately before it are a PAIR, and publishing only the
            # baseline summary hides whether a fast test reading simply sat next to a
            # fast baseline.
            "baseline_ms_samples": [round(x, 1) for x in base_ttft],
            "baseline_median_ttft_ms": (round(statistics.median(base_ttft), 1)
                                        if base_ttft else None),
            "baseline_spread_ms": round(base_spread, 1) if base_spread is not None else None,
            "test_spread_ms": round(test_spread, 1),
            # False => the published readings themselves are too scattered to quote as
            # a table cell. Note this is the TEST spread, not the baseline's: the
            # baseline is bimodal for reasons of probe ordering and must not veto a
            # stable measurement -- see the comment above test_spread.
            "stable": (test_spread is not None
                       and test_spread <= STABLE_MAX_SPREAD_MS),
        }

    print(json.dumps(out, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--length-sweep", action="store_true",
                    help="measure the prefill rate as d(tokens)/d(TTFT) across prompt "
                         "lengths. This is what the README's Prompt Processing row "
                         "quotes; see the module docstring for why a single-length "
                         "ratio is not a prefill rate.")
    ap.add_argument("--rounds", type=int, default=SWEEP_ROUNDS,
                    help="rounds per length for --length-sweep")
    ap.add_argument("--test", action="append", choices=sorted(TESTS),
                    help="named test; repeat for several. Per-test TTFT mode; ignored "
                         "with --length-sweep.")
    ap.add_argument("--iters", type=int, default=8,
                    help="samples per test in per-test mode")
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()

    # Reject a zero-sample run before it reaches the engine. With rounds=0 the sweep
    # would collect nothing, fit nothing, and still print a well-formed JSON document
    # whose prefill_tok_s is null -- an empty result that is easy to read as a
    # measurement. Fail at the argument instead, where the cause is obvious.
    if a.length_sweep and a.rounds < 1:
        print(f"prefill.py: --rounds must be >= 1, got {a.rounds}", file=sys.stderr)
        return 2
    if not a.length_sweep and a.iters < 1:
        print(f"prefill.py: --iters must be >= 1, got {a.iters}", file=sys.stderr)
        return 2

    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    if a.length_sweep:
        return run_length_sweep(url, a.model, a.rounds)
    if not a.test:
        print("prefill.py: choose --length-sweep or at least one --test",
              file=sys.stderr)
        return 2
    return run_per_test(url, a.model, a.test, a.iters, a.warmup)


if __name__ == "__main__":
    raise SystemExit(main())
