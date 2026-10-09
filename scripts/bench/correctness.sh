#!/usr/bin/env bash
# Correctness probe against a running endpoint.
#
# This is the check that matters. A graph-stride mismatch leaves throughput and
# acceptance counters looking completely healthy while the text is garbage
# ('duct', empty strings, CJK fragments), so a benchmark can never catch it --
# only a correctness probe can. It runs after every setup.sh and after any
# config change.
#
# Loop shape is a bash array, deliberately: an earlier version piped the prompts
# through `while IFS= read -r`, which split the one multi-line prompt
# ("def add(a, b):\n    return") into two prompts, tested something the measured
# arms never tested, and reported 12/18 on a perfectly healthy engine.
#
# The expected total is a hard gate. A run that finishes 12/18 is a harness bug,
# not a result, and must not be printable as a rate without the gate failing.
set -uo pipefail

PORT="${PORT:-8103}"; MNAME="${SERVED_NAME:-secondary}"
REPS="${REPS:-3}"
TIMEOUT_S="${TIMEOUT_S:-150}"

prompts=(
  "The capital of France is|Paris"
  "The capital of Germany is|Berlin"
  "The capital of Italy is|Rome"
  "Count up from one: 1 2|3 4 5"
  $'def add(a, b):\n    return|a + b'
)

timeout 10 curl -sf -m 8 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "[correctness] ABORT: /health not answering on $PORT"; exit 1; }

# printf '%s\n' "$t" | while read ... would split element 5. Count them first.
want_total=$(( ${#prompts[@]} * REPS ))
echo "[correctness] ${#prompts[@]} prompts x $REPS reps = $want_total probes (port $PORT, model $MNAME)"

good=0; total=0
for t in "${prompts[@]}"; do
  pr=${t%%|*}; want=${t##*|}
  for k in $(seq 1 "$REPS"); do
    out=$(PORT="$PORT" MNAME="$MNAME" PR="$pr" python3 - <<'PY' 2>/dev/null
import json, os, urllib.request
body = {"model": os.environ["MNAME"], "prompt": os.environ["PR"],
        "max_tokens": 24, "temperature": 0}
req = urllib.request.Request(f"http://127.0.0.1:{os.environ['PORT']}/v1/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
print(json.loads(urllib.request.urlopen(req, timeout=140).read().decode())
      ["choices"][0]["text"][:60])
PY
)
    total=$((total+1))
    if grep -qF -- "$want" <<<"$out"; then
      good=$((good+1))
    else
      echo "[correctness]   MISS rep=$k prompt=$(printf %q "$pr") want=$(printf %q "$want") got=$(printf %q "$out")"
    fi
  done
done

echo "[correctness] $good/$total correct"
# Two separate failures: wrong output, and a loop that did not run the probes it
# claims to have run.
if [[ $total -ne $want_total ]]; then
  echo "[correctness] FAIL: ran $total probes but expected $want_total -- harness bug, not an engine result"
  exit 1
fi
if [[ $good -ne $total ]]; then
  echo "[correctness] FAIL: $good/$total -- suspect a graph stride/buffer mismatch before suspecting the drafter"
  exit 1
fi
echo "[correctness] PASS"
