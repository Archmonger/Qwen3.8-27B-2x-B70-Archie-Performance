#!/usr/bin/env python3
"""Reproduce the canonical suite metric: median generated-token throughput for
tokens 1-100 after TTFT, across the 12 fixed cold prompts in
`data/realistic-suite-v1.json` (each sent exactly once, once per run).

Usage:  PORT=8103 SERVED_NAME=secondary ./suite.py
        ./suite.py [port] [model]        (positional form also accepted)
        SUITE_JSON=/path/to/suite.json ./suite.py

Counting rule, and why it matters: the request asks for `return_token_ids`, so
one SSE event == one generated token. A multi-token event (a speculative accept)
advances the counter by len(token_ids) but shares one timestamp. Counting raw SSE
*text* events instead understates a spec-decoding server badly -- 7.6 tok/s vs
38.1 tok/s on the same request stream -- and that trap has already produced a
false parity conclusion on this hardware.

Engine-log "Avg generation throughput" is a different (aggregate) metric and is
not comparable to this one.

The prompt list is read from the JSON, not embedded here, so the file is the
single source of truth and a swapped suite cannot silently diverge from the
harness. The 86.93 / 94.58 figures quoted inside the JSON's own description come
from a DIFFERENT checkpoint on a patched vLLM fork and are not a parity target
for this deployment -- see "Why two published numbers are not comparable" in the
README.
"""
import json, os, statistics, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT") or (sys.argv[1] if len(sys.argv) > 1 else 8103))
MODEL = os.environ.get("SERVED_NAME") or (sys.argv[2] if len(sys.argv) > 2 else "secondary")
SUITE_JSON = os.environ.get(
    "SUITE_JSON", os.path.join(HERE, "..", "..", "data", "realistic-suite-v1.json"))
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 512))


def load_prompts(path):
    with open(path) as f:
        doc = json.load(f)
    prompts = [p["prompt"] if isinstance(p, dict) else p for p in doc["prompts"]]
    if len(prompts) < 1:
        raise SystemExit(f"no prompts in {path}")
    print(f"suite {doc.get('suite_id', os.path.basename(path))} "
          f"({len(prompts)} prompts, each sent once)")
    return prompts


def one(prompt: str):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        # Thinking off is what every recorded suite number used; leaving the
        # server default (thinking + reasoning_effort xhigh) in would measure a
        # different distribution than the one the reference rows came from.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(BASE, data=body, headers={
        "Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    n = 0
    t100 = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            tids = (delta.get("token_ids") or ch.get("token_ids")
                    or d.get("token_ids"))
            if tids:
                k = len(tids)
            elif delta.get("content"):
                k = 1
            else:
                continue
            if t_first is None:
                t_first = time.perf_counter()
            for _ in range(k):
                n += 1
                if n == 100 and t100 is None:
                    t100 = time.perf_counter()
    end = time.perf_counter()
    if t_first is None:
        return None
    stop = t100 if t100 is not None else end
    span = max(stop - t_first, 1e-9)
    counted = min(n, 100) if t100 is not None else n
    return {
        "tokens": n,
        "ttft_ms": round((t_first - t0) * 1000, 1),
        "tok_s_1_100_after_ttft": round(counted / span, 2) if counted else 0.0,
        "tok_s_after_ttft_full": round(n / max(end - t_first, 1e-9), 2),
    }


def main():
    prompts = load_prompts(SUITE_JSON)
    rows = []
    for p in prompts:
        try:
            r = one(p)
        except Exception as e:
            # A dropped prompt must be visible, not silently averaged away: a
            # 12-prompt median over 9 rows reads exactly like a 12-prompt median.
            print(f"  ERROR on {p[:44]!r}: {type(e).__name__}: {e}")
            continue
        if r:
            rows.append(r)
            print(f"{r['tok_s_1_100_after_ttft']:8.2f} tok/s (1-100)  "
                  f"{r['tok_s_after_ttft_full']:8.2f} full  ttft {r['ttft_ms']:7.1f} ms  "
                  f"tokens {r['tokens']:4d}  | {p[:44]}")
        else:
            print(f"  no tokens streamed for {p[:44]!r}")

    if len(rows) < len(prompts):
        print(f"\nINCOMPLETE: {len(rows)} of {len(prompts)} prompts produced samples. "
              "Do not quote the median below as the suite metric.", file=sys.stderr)

    if not rows:
        return 1

    def s(key):
        v = sorted(r[key] for r in rows if r[key])
        if not v:
            return {"count": 0}
        return {"count": len(v), "median": statistics.median(v),
                "mean": round(statistics.fmean(v), 2), "min": v[0], "max": v[-1]}

    print("\n=== SUMMARY ===")
    for k in ("tok_s_1_100_after_ttft", "tok_s_after_ttft_full", "ttft_ms"):
        print(f"{k:26s} {s(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
