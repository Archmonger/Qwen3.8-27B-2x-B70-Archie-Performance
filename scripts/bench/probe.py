#!/usr/bin/env python3
"""Replicate PR #620's "isolated C1" probe protocol exactly (stdlib only).

PR recipe metric: temp=0, pp=0, n=16 generated tokens, prompt "1 2" (~3
tokens), 10 measured requests after warmup. Published: 72.2 tok/s median
(85.9 peak); independent replication on 2x B70: 71.20 median / 70.98 mean /
71.54 peak with 98.8% draft-token acceptance.

Uses return_token_ids so one SSE event == one generated token; a multi-token
event (speculative accept) advances the token counter by len(token_ids) but
shares one timestamp, matching suite.py's counting rule.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request


def one(base_url: str, model: str, prompt: str, max_tokens: int, path: str):
    if path == "chat":
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "return_token_ids": True,
            "prompt": prompt,
        }
    req = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    stamps = []
    events = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            choices = ev.get("choices") or [{}]
            ch = choices[0]
            delta = ch.get("delta") or {}
            tids = (delta.get("token_ids") or ch.get("token_ids")
                    or ev.get("token_ids"))
            if tids:
                k = len(tids)
            else:
                text = ch.get("text")
                if text is None:
                    text = delta.get("content")
                if not text:
                    continue
                k = 1
            events += 1
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            for _ in range(k):
                stamps.append(now)
    if len(stamps) < 2:
        return None
    span = stamps[-1] - stamps[0]
    return {
        "n": len(stamps),
        "events": events,
        "ttft": ttft,
        "tok_s_after_ttft": (len(stamps) - 1) / span if span > 0 else 0.0,
        "tok_s_incl_ttft": len(stamps) / (stamps[-1] - t0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8103/v1/completions")
    ap.add_argument("--path", default="completions", choices=["completions", "chat"])
    ap.add_argument("--model", default="secondary")
    ap.add_argument("--prompt", default="1 2")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    for _ in range(args.warmup):
        try:
            one(args.url, args.model, args.prompt, args.max_tokens, args.path)
        except Exception as exc:
            print(f"warmup error: {exc}", file=sys.stderr)

    rows = []
    for _ in range(args.iters):
        try:
            r = one(args.url, args.model, args.prompt, args.max_tokens, args.path)
        except Exception as exc:
            print(f"measured error: {exc}", file=sys.stderr)
            continue
        if r:
            rows.append(r)
    if not rows:
        print("no usable samples")
        return 1

    aft = [r["tok_s_after_ttft"] for r in rows]
    incl = [r["tok_s_incl_ttft"] for r in rows]
    ttfts = [r["ttft"] for r in rows]
    print(json.dumps({
        "label": args.label,
        "url": args.url,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "samples": len(rows),
        "tokens_per_req": rows[-1]["n"],
        "events_per_req": rows[-1]["events"],
        "median_tok_s_after_ttft": round(statistics.median(aft), 2),
        "mean_tok_s_after_ttft": round(statistics.mean(aft), 2),
        "peak_tok_s_after_ttft": round(max(aft), 2),
        "median_tok_s_incl_ttft": round(statistics.median(incl), 2),
        "median_ttft_ms": round(1000 * statistics.median(ttfts), 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
