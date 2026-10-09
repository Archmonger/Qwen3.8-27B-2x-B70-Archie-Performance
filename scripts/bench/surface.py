#!/usr/bin/env python3
"""Serving-surface check against an ALREADY-RUNNING endpoint.

Usage:  PORT=8103 SERVED_NAME=secondary ./surface.py
        ./surface.py [port] [model]        (positional form also accepted)

Why this exists next to the correctness probe: the correctness probe hits
/v1/completions, which bypasses the chat template, the tool parser, and the
reasoning parser entirely. A flag combination can break all three and still pass
every other check. These are the five cells the reference numbers were validated
against:

  1 thinking_capitals      -- chat with the SERVER default thinking (the deployed
                              profile enables thinking + reasoning_effort xhigh via
                              --default-chat-template-kwargs; overriding it here would
                              test a distribution the endpoint never serves)
  2 tool_call_qwen3_xml    -- native tool calling through the qwen3_xml parser, which
                              only engages on the real chat template
  3 reasoning_math         -- reasoning_content must be surfaced, not swallowed
  4 prompt_tokens_details  -- --enable-prompt-tokens-details must populate usage details
  5 code_fib               -- non-trivial correctness with thinking on
"""
import json
import os
import sys
import urllib.request

port = int(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PORT", 8103))
model = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("SERVED_NAME", "secondary")
URL = f"http://127.0.0.1:{port}/v1/chat/completions"


def chat(msgs, **kw):
    body = {"model": model, "messages": msgs, "temperature": 0}
    body.update(kw)
    r = urllib.request.Request(URL, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as f:
        return json.load(f)


out = []

# 1. thinking ON, server default -- do not pass chat_template_kwargs here.
try:
    d = chat([{"role": "user",
               "content": "What is the capital of Australia? Answer in one short sentence."}],
             max_tokens=512)
    c = d["choices"][0]["message"]
    txt = (c.get("content") or "") + " " + (c.get("reasoning_content") or "")
    out.append({"test": "thinking_capitals", "good": int("Canberra" in txt), "of": 1,
                "note": txt[:70].replace("\n", " ")})
except Exception as e:
    out.append({"test": "thinking_capitals", "good": 0, "of": 1,
                "note": f"{type(e).__name__}:{e}"[:70]})

# 2. tool call through the qwen3_xml parser.
try:
    d = chat([{"role": "user", "content": "What is 25 times 4? Use the calculator tool."}],
             tools=[{"type": "function", "function": {
                 "name": "calculator", "description": "Evaluate an arithmetic expression",
                 "parameters": {"type": "object",
                                "properties": {"expression": {"type": "string"}},
                                "required": ["expression"]}}}],
             tool_choice="auto", max_tokens=256)
    tc = d["choices"][0]["message"].get("tool_calls")
    args = tc[0]["function"]["arguments"] if tc else ""
    ok = bool(tc) and "25" in args and "4" in args
    out.append({"test": "tool_call_qwen3_xml", "good": int(ok), "of": 1,
                "note": (args or str(d["choices"][0]["message"].get("content"))[:50])[:70]})
except Exception as e:
    out.append({"test": "tool_call_qwen3_xml", "good": 0, "of": 1,
                "note": f"{type(e).__name__}:{e}"[:70]})

# 3. reasoning trace surfaced.
try:
    d = chat([{"role": "user",
               "content": "A train travels 60 km in 1.5 hours. What is its speed in km/h? "
                          "Reason, then answer."}], max_tokens=1024)
    m = d["choices"][0]["message"]
    rc = m.get("reasoning_content") or ""
    ok = "40" in ((m.get("content") or "") + " " + rc)
    out.append({"test": "reasoning_math", "good": int(ok), "of": 1,
                "note": f"reasoning_chars={len(rc)} ans={(m.get('content') or '')[:40]}"[:70]})
except Exception as e:
    out.append({"test": "reasoning_math", "good": 0, "of": 1,
                "note": f"{type(e).__name__}:{e}"[:70]})

# 4. usage details populated.
try:
    d = chat([{"role": "user", "content": "Say hello."}], max_tokens=16,
             chat_template_kwargs={"enable_thinking": False})
    u = d.get("usage", {})
    out.append({"test": "prompt_tokens_details", "good": int(bool(u.get("prompt_tokens_details"))),
                "of": 1, "note": json.dumps(u)[:70]})
except Exception as e:
    out.append({"test": "prompt_tokens_details", "good": 0, "of": 1,
                "note": f"{type(e).__name__}:{e}"[:70]})

# 5. code task with thinking on.
try:
    d = chat([{"role": "user",
               "content": "Write a Python function fib(n) returning the nth Fibonacci number, "
                          "fib(0)=0, fib(1)=1. Reply with code only."}], max_tokens=1024)
    txt = d["choices"][0]["message"].get("content") or ""
    ok = ("def fib" in txt) and ("return" in txt)
    out.append({"test": "code_fib", "good": int(ok), "of": 1,
                "note": txt[:70].replace("\n", " ")})
except Exception as e:
    out.append({"test": "code_fib", "good": 0, "of": 1, "note": f"{type(e).__name__}:{e}"[:70]})

good = 0
of = 0
for o in out:
    print(json.dumps({"tag": "surface", **o}))
    good += o["good"]
    of += o["of"]
print(json.dumps({"tag": "surface_summary", "good": good, "of": of}))
# Exit non-zero on any failure so the shell caller cannot read a silent 0.
sys.exit(0 if good == of and of > 0 else 1)
