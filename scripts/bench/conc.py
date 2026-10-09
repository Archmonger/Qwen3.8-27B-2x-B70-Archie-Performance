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
arm runs second.

--max-tokens IS PART OF THE METRIC, NOT A KNOB. aggregate_tok_s divides counted
completion_tokens by wall time, and wall time also pays for prefill. With the fixed
~243-token prompt here, that uncounted prefill cost is amortised over max_tokens, so
the same boot at 96 reads materially slower at high concurrency than at 256 (measured
on one boot, alternating order: conc10 193-203 at 96 vs 229-232 at 256, while conc1
moved only 4% because prefix caching makes prefill nearly free for a single stream).
This is why the depression once looked like a host fault: it moved the batched cells
and left single-stream alone. So the number is only comparable to another number
taken at the same max_tokens, and max_tokens is therefore printed with every cell and
defaults to 256, the value the recorded README column used. Deviating requires an
explicit --confirm-noncanonical, which exists so exploration stays possible without
letting an accidental default produce an unattributable headline number.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

_PROMPT_FILE = os.environ.get(
    "PROMPT_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "prompt512.txt"))
with open(_PROMPT_FILE) as _f:
    PROMPT = _f.read().strip()

# The value every recorded number in the README was produced with. It is a metric
# definition, not a default to be tuned -- see the module docstring. Both scripted
# callers (scripts/bench/run-all.sh, scripts/soak.sh) pass exactly this, so only a
# hand-run conc.py can drift from it, which is precisely what happened once.
CANONICAL_MAX_TOKENS = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=CANONICAL_MAX_TOKENS,
                    help=f"tokens/request; part of the metric definition. "
                         f"Default and canonical value is {CANONICAL_MAX_TOKENS}; any "
                         f"other value also needs --confirm-noncanonical.")
    ap.add_argument("--confirm-noncanonical", action="store_true",
                    help="required to run at a --max-tokens other than "
                         f"{CANONICAL_MAX_TOKENS}; results then must not be compared "
                         "against the recorded column")
    ap.add_argument("--cells", default="1:6,8:24",
                    help="conc:total pairs, comma separated")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.max_tokens != CANONICAL_MAX_TOKENS and not a.confirm_noncanonical:
        print(
            f"conc.py: --max-tokens {a.max_tokens} is not the canonical "
            f"{CANONICAL_MAX_TOKENS}.\n"
            f"  aggregate_tok_s moves ~15% on this parameter, so a number taken at "
            f"any\n  other value is NOT comparable to the recorded README column and "
            f"must not be\n  quoted as one. This is the trap that once read as a "
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

    def gen(i):
        body = {
            "model": a.model,
            "messages": [{"role": "user", "content": f"({i % 8}) {PROMPT}"}],
            "max_tokens": a.max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
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
    # ended up being compared against a 256-token record.
    tag = f"max_tokens={a.max_tokens}" + (
        "" if a.max_tokens == CANONICAL_MAX_TOKENS else " (NON-CANONICAL)")
    if a.json:
        print(json.dumps({"max_tokens": a.max_tokens,
                          "canonical_max_tokens": a.max_tokens == CANONICAL_MAX_TOKENS,
                          "cells": out}))
    else:
        print(f"  [{tag}] cells={a.cells}")
        for k, v in out.items():
            print(f"  {k}: {v['aggregate_tok_s']:6.2f} tok/s  "
                  f"({v['tokens']} toks in {v['wall_s']}s, median req {v['median_req_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
