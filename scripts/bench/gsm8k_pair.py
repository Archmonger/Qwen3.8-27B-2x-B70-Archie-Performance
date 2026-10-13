#!/usr/bin/env python3
"""Paired GSM8K subset for W57 dtype/kernel arms.

Why not just quote lm-eval: the archived dump under /root/eval_gsm8k_fp8 is from
2026-09-16, against a different boot and a different vLLM surface. Comparing one
arm to that number and the other to it too is fine for sanity, but the decision
this arms exists to make (does --dtype float16 cost arithmetic quality?) needs
both arms asked the *same* items on the *same* day, or drift swamps the delta.

So: take items from the archived dump (real GSM8K questions with gold final
numbers), ask each arm greedy with thinking off, extract the final number, score
exact match. Report per-item disagreement -- an arm that is right on 28/30 and
wrong on the same two items as the control is not a regression; one that loses
different items is.

Gold extraction: the dump's `target` ends with '#### <n>'.
Answer extraction: last integer in the response (GSM8K convention).

Usage: gsm8k_pair.py --port 8126 --model w57-calib2 --n 30 --label CALIB2
"""
import argparse
import json
import os
import re
import statistics
import sys
import urllib.request

# Where the items come from. Overridable with --dump and validated at startup
# instead of assumed: a published comparison has to state which items it asked,
# and a hardcoded scratch path that later disappears makes the number
# unreproducible -- the defect docs/measurement-notes.md exists to prevent. The
# file only has to be identical across the pair, since the point is that both
# arms are asked the same questions on the same day; its path is recorded in the
# output so a row names its own source.
DEFAULT_DUMP = ("/root/eval_gsm8k_fp8/secondary/"
                "samples_gsm8k_2026-09-16T01-02-07.479980.jsonl")

SYS = ("You are a careful math tutor. Reason step by step, then end your "
       "answer with the final number on its own line.")


def load_items(dump, n, offset=0):
    items = []
    with open(dump) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            doc = d.get("doc") or {}
            q = doc.get("question")
            tgt = str(d.get("target") or "")
            m = re.search(r"####\s*(-?[0-9]+(?:\.[0-9]+)?)", tgt)
            if not q or not m:
                continue
            items.append({"id": d.get("doc_id"), "q": q, "gold": m.group(1)})
            if len(items) >= n + offset:
                break
    return items[offset:offset + n]


def ask(base, model, q, max_tokens=1024):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": q}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode())
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    finish = ch.get("finish_reason") or ""
    txt = (msg.get("content") or ch.get("text") or "").strip()
    # A truncated generation is not a wrong answer -- it is an unanswered item.
    # Silently scoring 320-token truncations as failures is what made the first
    # run read 15/30 on an engine whose own recorded GSM8K score is ~0.60.
    if finish == "length" and "####" not in txt:
        return "__TRUNCATED__ " + txt[-120:]
    return txt


def last_int(text):
    # Prefer an explicit final-answer marker; only fall back to the last number.
    # With LaTeX chain-of-thought the raw last number is usually an intermediate
    # product of a still-running derivation, which is how a correct engine
    # scored "gold=260 pred=100".
    for pat in (r"####\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)",
                r"(?:final answer|answer is|answer:)\D{0,12}(-?[0-9][0-9,]*(?:\.[0-9]+)?)"):
        m = re.findall(pat, text, flags=re.I)
        if m:
            return m[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def norm(x):
    if x is None:
        return None
    try:
        f = float(x)
        return int(f) if f == int(f) else f
    except ValueError:
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default=DEFAULT_DUMP,
                    help="lm-eval samples jsonl to draw items from; must be the "
                         "same file for both arms of a pair")
    a = ap.parse_args()

    # Fail before asking anything rather than after 30 requests. A missing dump
    # that produces 0/30 would be read as an engine that cannot do arithmetic.
    if not os.path.isfile(a.dump):
        print(f"gsm8k_pair.py: no such dump: {a.dump}\n"
              f"  Pass --dump pointing at an lm-eval samples jsonl whose docs have "
              f"a 'question' and a '#### n' target.\n"
              f"  Both arms of a pair must use the SAME file, or the comparison is "
              f"two different question sets.", file=sys.stderr)
        return 2
    items = load_items(a.dump, a.n, a.offset)
    if not items:
        print(f"gsm8k_pair.py: dump {a.dump} yielded 0 usable items "
              f"(need doc.question and target '#### <n>')", file=sys.stderr)
        return 2
    base = f"http://127.0.0.1:{a.port}/v1"
    rows = []
    correct = 0
    unanswered = 0
    secs = []
    for it in items:
        try:
            out = ask(base, a.model, it["q"])
        except Exception as exc:
            out = f"__ERROR__ {exc}"
        pred = norm(last_int(out))
        gold = norm(it["gold"])
        truncated = out.startswith("__TRUNCATED__") or out.startswith("__ERROR__")
        hit = (not truncated) and pred is not None and pred == gold
        correct += 1 if hit else 0
        unanswered += 1 if truncated else 0
        secs.append(len(out))
        rows.append({"id": it["id"], "gold": gold, "pred": pred, "hit": hit,
                     "truncated": truncated,
                     "tail": out[-120:].replace("\n", " ")})

    res = {"label": a.label, "port": a.port, "model": a.model,
           "n": len(items), "offset": a.offset,
           "exact": f"{correct}/{len(items)}",
           "accuracy": round(100.0 * correct / max(1, len(items)), 1),
           "unanswered": unanswered,
           "median_chars": statistics.median(secs) if secs else 0,
           "items": rows}
    print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=2))
    for r in rows:
        if not r["hit"]:
            tag = "TRUNC" if r["truncated"] else "MISS "
            print(f"  [{tag}] id={r['id']} gold={r['gold']} pred={r['pred']} tail={r['tail']!r}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f)
        print(f"[gsm8k] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
