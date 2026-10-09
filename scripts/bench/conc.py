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
"""
import argparse
import json
import os
import statistics
import threading
import time
import urllib.request

_PROMPT_FILE = os.environ.get(
    "PROMPT_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "prompt512.txt"))
with open(_PROMPT_FILE) as _f:
    PROMPT = _f.read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--cells", default="1:6,8:24",
                    help="conc:total pairs, comma separated")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
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
    if a.json:
        print(json.dumps(out))
    else:
        for k, v in out.items():
            print(f"  {k}: {v['aggregate_tok_s']:6.2f} tok/s  "
                  f"({v['tokens']} toks in {v['wall_s']}s, median req {v['median_req_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
