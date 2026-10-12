#!/usr/bin/env python3
"""Exact-answer quality battery for W57 dtype/kernel arms.

Why this exists: the 15-probe correctness check (capitals, "1 2"->"3 4 5",
`def add`) has been passing on every arm this session, including an arm that
lost a third of its single-stream throughput. It cannot see a quality change at
all. A dtype or kernel-selection change is exactly the class of edit that can
produce output that is fluent, on-topic, and *wrong* -- which is the failure
mode that makes a fast checkpoint worthless. So an arm must clear a battery of
verifiable answers, not just a fluency smoke test.

Each case is (prompt, required-substring-list). All must appear. Temperature 0,
thinking disabled, chat surface (the instruction-tuned surface is the one that
follows "answer with only the number"; raw completions do not, and scored a
false fail on CALIB1).

Determinism is reported as its own axis: every case is asked TWICE on the same
boot. A spec-decode or graph change can make greedy output non-reproducible
without making it wrong on average, and that instability is the leading
indicator of the stride/graph class of defect this host has been burned by
before.

Usage: quality.py --port 8125 --model w57-fp16full-b2 [--json]
"""
import argparse
import json
import statistics
import time
import urllib.request

# (prompt, [substrings that must all appear])
CASES = [
    ("Compute the sum of i*i for i in range(4). Answer with only the number.", ["14"]),
    ("What is 17 multiplied by 23? Answer with only the number.", ["391"]),
    ("A train covers 120 km in 2 hours, then 180 km in 3 hours. What is its "
     "average speed in km/h over the whole trip? Answer with only the number.", ["60"]),
    ("How many days are there in 2024? Answer with only the number.", ["366"]),
    ("What is the chemical formula for water? Answer with only the formula.", ["H2O"]),
    ("Name the three primary colours of additive (light) colour mixing, "
     "comma separated.", ["red", "green", "blue"]),
    ("In what year did World War II end? Answer with only the year.", ["1945"]),
    ("Who wrote the novel '1984'? Answer with only the surname.", ["Orwell"]),
    ("What is the binary representation of the decimal number 10? "
     "Answer with only the digits.", ["1010"]),
    ("Solve for x: 3x + 7 = 22. Answer with only the number.", ["5"]),
    ("List the first five prime numbers, comma separated.", ["2", "3", "5", "7", "11"]),
    ("What is the capital of Japan? Answer with only the city.", ["Tokyo"]),
    ("How many bits are in one byte? Answer with only the number.", ["8"]),
    ("What does the acronym CPU stand for? Spell it out in full.", ["central", "processing", "unit"]),
    ("Which element has the chemical symbol Fe? Answer with only the element name.", ["iron"]),
    ("What is 2 to the power of 10? Answer with only the number.", ["1024"]),
    ("Balance the description: in Python, which single keyword begins a "
     "function definition? Answer with only the keyword.", ["def"]),
    ("A rectangle is 8 units by 5 units. What is its area? Answer with only the number.", ["40"]),
    ("What is the largest planet in our solar system? Answer with only the name.", ["Jupiter"]),
    ("How many continents are there on Earth? Answer with only the number.", ["7"]),
]


def ask(base, model, prompt, max_tokens=64):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode())
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    return (msg.get("content") or ch.get("text") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"

    total = 0
    good = 0
    unstable = 0
    rows = []
    ttfts = []
    for prompt, wants in CASES:
        outs = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            try:
                out = ask(base, a.model, prompt)
            except Exception as exc:
                out = f"__ERROR__ {exc}"
            ttfts.append(time.perf_counter() - t0)
            outs.append(out)
        low = outs[0].lower()
        # A transport error must never score as a hit. On the first run the
        # "area = 40" case passed against the string "__ERROR__ HTTP Error 404:
        # Not Found" because that string contains "40" -- a vacuous pass that
        # would have hidden a dead endpoint behind a green battery. Likewise
        # "identical error strings" must not count as reproducible.
        dead = outs[0].startswith("__ERROR__") or not outs[0].strip()
        hit = (not dead) and all(w.lower() in low for w in wants)
        stable = (not dead) and all(o == outs[0] for o in outs)
        total += 1
        good += 1 if hit else 0
        unstable += 0 if stable else 1
        rows.append({"prompt": prompt[:56], "want": wants,
                     "got": outs[0][:90], "hit": hit, "stable": stable})

    print(json.dumps({
        "port": a.port, "model": a.model,
        "exact_correct": f"{good}/{total}",
        "nonreproducible": f"{unstable}/{total}",
        "median_answer_s": round(statistics.median(ttfts), 3),
        "cases": rows,
    }, indent=2) if a.json else "")
    for r in rows:
        flag = "ok  " if r["hit"] else "MISS"
        unst = "" if r["stable"] else "  [NON-REPRODUCIBLE]"
        print(f"  [{flag}] {r['prompt']!r} want={r['want']} got={r['got']!r}{unst}")
    print(f"[quality] exact_correct {good}/{total}  "
          f"non-reproducible {unstable}/{total}  "
          f"median_answer_s {statistics.median(ttfts):.3f}")
    return 0 if good == total and unstable == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
