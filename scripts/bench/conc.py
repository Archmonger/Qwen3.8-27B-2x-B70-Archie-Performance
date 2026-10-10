#!/usr/bin/env python3
"""Concurrency-aware aggregate throughput against any endpoint.

Takes --port and --model because a /health 200 does not identify an engine: an A/B
runs several containers on ephemeral ports under distinct served names, and a probe
hardwired to 8103/"secondary" would silently measure whichever engine happened to be
up.

Why concurrency is the metric that decides these comparisons: the endpoint is shared
and capped at MAX_NUM_SEQS. Single-stream tok/s has repeatedly disagreed with what the
box does under load, and decisions taken on conc=8 aggregate missed a 38% batched loss
because the conc=10 step is the one that crosses the graph ceiling (see
docs/cudagraph-stride-defect.md). Batching is exactly where DSpark's accepted tokens
pay off and exactly where a captured graph stops adding work per step, so a config
that wins on a single stream can lose the moment requests batch.

Always include a cell at the MAX_NUM_SEQS cap. Anything below it may not be the state
a busy engine is ever in.

Prompts vary by an (i % 8) prefix so prefix caching cannot make later requests
artificially cheap -- with caching on, a constant prompt otherwise flatters whichever
arm runs second. Keep it even though it is currently inert here: on this deployment
`cached_tokens` stays 0 until a prompt fills at least two full 2048-token KV blocks.
Measured directly -- identical repeats report 0 at 296, 347, 1292 and even 2572 prompt
tokens, and 4096 at 6412, which is the pattern of whole blocks minus the one holding the
final token. The variation therefore costs nothing today and is the only thing standing
between this harness and a flattering-second-arm result once prompts grow past ~4096
tokens. Trust the per-request cached_tokens field for this, NOT the engine's aggregate
"Prefix cache hit rate" log line: that figure is cumulative over the container's life
and climbs off whatever long-prompt traffic the box has ever seen (it read up to 12.3%
after a session of length sweeps), so on a probed host it is nonzero even while every
prompt this harness sends still reports cached_tokens=0.

--max-tokens IS PART OF THE METRIC, NOT A KNOB. aggregate_tok_s divides counted
completion_tokens by wall time, and wall time also pays for prefill. The prompt here is
a fixed few-hundred-token block (server-reported prompt_tokens, not a word count), so
that uncounted prefill cost is amortised over max_tokens, and the same boot at 96 reads
materially slower at high concurrency than at 256 -- measured on one boot with
alternating order: conc10 193-203 at 96 vs 229-232 at 256, while conc1 moved only ~4%.
That asymmetry is what made the depression look like a host fault: it moved the batched
cells and left single stream alone.

Do not repeat the explanation that used to be written here. It attributed the asymmetry
to prefix caching making single-stream prefill nearly free, and that is falsified:
prefix caching does not engage at these prompt sizes at all (identical repeats return
cached_tokens=0), so it cannot be making anything free in either cell. What is
established is the size of the effect -- ~15% on the batched cells -- which is the whole
justification for the guard. Why single stream is nearly immune while the capped cells
move is NOT established; treat it as an open question, and do not cite a cause for it.

So the number is only comparable to another number taken at the same max_tokens, and
max_tokens is therefore printed with every cell and defaults to 256, the value the
historical recorded numbers used. Deviating requires an explicit
--confirm-noncanonical, which exists so exploration stays possible without letting an
accidental default produce an unattributable headline number.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")

# Named tests, each one (prompt file, max_tokens) pair. A test is a metric
# DEFINITION, not a suggestion: the prompt and the generation budget together decide
# what the number means, so both are pinned here next to each other and a run that
# deviates from either has to say so explicitly. `max_tokens` is not a knob --
# aggregate_tok_s moves ~15% between 96 and 256 on the same boot (module docstring),
# so a per-test value is only meaningful because it is recorded with the number.
#
# The budget is the prompt's natural completion length, measured rather than guessed:
# each prompt was run at a generous cap with temperature=0 and the cap chosen just
# above where it stops on its own. Counting stops at 372 tokens; Planning and Prose
# stop around 1845; Coding is still generating at 2048, so its budget is a cap and not
# a completion -- recorded that way, because a truncated generation is a different
# measurement and the table must not imply otherwise.
TESTS = {
    "planning": {"prompt": "prompt-planning.txt", "max_tokens": 2048},
    "counting": {"prompt": "prompt-counting.txt", "max_tokens": 512},
    "prose":    {"prompt": "prompt-prose.txt",    "max_tokens": 2048},
    "coding":   {"prompt": "prompt-coding.txt",   "max_tokens": 2048},
}

# The value every number recorded before per-test prompts existed was produced with.
# It stays canonical for a bare `conc.py` run with no --test, so the historical
# recorded column remains reproducible and the guard keeps a meaning for callers that
# predate the test table. Both scripted callers pass exactly this when --test is absent.
CANONICAL_MAX_TOKENS = 256


def load_prompt(name):
    """Read a test prompt from data/. Missing file is a hard error: silently
    substituting a default would measure something else and still print a number."""
    path = os.path.join(_DATA, name)
    with open(path) as fh:
        return fh.read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", choices=sorted(TESTS), default=None,
                    help="named test from the TESTS table; fixes BOTH the prompt and "
                         "--max-tokens. Without it, the pre-test behaviour is kept: "
                         "prompt-planning.txt at the canonical max_tokens, which is "
                         "what the historical recorded numbers used.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="tokens/request; part of the metric definition. Defaults to "
                         "the test's pinned value, or the canonical "
                         f"{CANONICAL_MAX_TOKENS} with no --test. Any other value also "
                         "needs --confirm-noncanonical.")
    ap.add_argument("--confirm-noncanonical", action="store_true",
                    help="required to run at a --max-tokens other than the pinned "
                         "value; results then must not be compared against the "
                         "recorded table")
    ap.add_argument("--cells", default="1:6,8:24",
                    help="conc:total pairs, comma separated")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    # Resolve the metric definition first: the prompt and the budget come from one
    # table, so a named test cannot be run with a prompt that does not belong to it.
    if a.test:
        spec = TESTS[a.test]
        prompt_file, canonical = spec["prompt"], spec["max_tokens"]
    else:
        prompt_file, canonical = "prompt-planning.txt", CANONICAL_MAX_TOKENS
    max_tokens = canonical if a.max_tokens is None else a.max_tokens

    if max_tokens != canonical and not a.confirm_noncanonical:
        print(
            f"conc.py: --max-tokens {max_tokens} is not the canonical "
            f"{canonical}"
            + (f" for test {a.test!r}.\n" if a.test else ".\n")
            + f"  aggregate_tok_s moves ~15% on this parameter, so a number taken at "
              f"any\n  other value is NOT comparable to the recorded table and must "
              f"not be\n  quoted as one. This is the trap that once read as a "
              f"'13% host-side\n  regression' on a boot that was provably correct.\n"
              f"  Exploring?  add --confirm-noncanonical\n"
              f"  Measuring? use the default (drop --max-tokens) or run "
              f"scripts/bench/run-all.sh",
            file=sys.stderr)
        # 3, deliberately not 2: the warmup-failure path below also returns 2, and
        # a caller must be able to tell "the guard rejected this run" from "the
        # engine was unreachable". Same reason soak.sh keeps 124 and 143 apart.
        return 3
    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    prompt = load_prompt(prompt_file)

    def gen(i):
        body = {
            "model": a.model,
            "messages": [{"role": "user", "content": f"({i % 8}) {prompt}"}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        # 1800 s, not the 600 s that sufficed when every cell generated 256 tokens:
        # the per-test budgets are up to 2048 tokens, and at conc=10 a request can
        # legitimately sit behind several waves before it finishes. A timeout that
        # fires inside a valid cell would look like an engine fault.
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
        dt = time.perf_counter() - t0
        return d["usage"]["completion_tokens"], dt

    def run(conc, total):
        lock = threading.Lock()
        done = [0]
        toks = [0]
        per = []
        t0 = time.perf_counter()

        def worker(idx):
            while True:
                with lock:
                    if done[0] >= total:
                        return
                    done[0] += 1
                    k = done[0]
                n, dt = gen(k + idx * 97)
                with lock:
                    toks[0] += n
                    per.append(dt)

        ths = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.perf_counter() - t0
        return toks[0] / wall, wall, per

    # One throwaway request: Triton JITs layer_norm / postprocess_mamba /
    # _prepare_dflash_inputs on the first inferences, which would otherwise be billed to the
    # first cell.
    try:
        gen(0)
    except Exception as e:
        print(f"WARMUP_FAILED {type(e).__name__}: {e}")
        return 2

    out = {}
    for cell in a.cells.split(","):
        conc, total = (int(x) for x in cell.split(":"))
        agg, wall, per = run(conc, total)
        out[f"conc{conc}"] = {"aggregate_tok_s": round(agg, 2),
                              "wall_s": round(wall, 1),
                              "tokens": int(agg * wall),
                              "median_req_s": round(statistics.median(per), 2) if per else None}
    # Echo max_tokens on every cell. It changes the number by ~15%, so a result
    # without it attached cannot be audited later -- which is how a 96-token run
    # ended up being compared against a 256-token record. The test name and prompt
    # file are echoed for the same reason: a rep file that does not say which prompt
    # produced it can be filed against the wrong row.
    tag = f"max_tokens={max_tokens}" + (
        "" if max_tokens == canonical else " (NON-CANONICAL)")
    if a.json:
        print(json.dumps({"test": a.test, "prompt": prompt_file,
                          "max_tokens": max_tokens,
                          "canonical_max_tokens": max_tokens == canonical,
                          "cells": out}))
    else:
        print(f"  [{tag}] test={a.test or '(default planning prompt)'} cells={a.cells}")
        for k, v in out.items():
            print(f"  {k}: {v['aggregate_tok_s']:6.2f} tok/s  "
                  f"({v['tokens']} toks in {v['wall_s']}s, median req {v['median_req_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
