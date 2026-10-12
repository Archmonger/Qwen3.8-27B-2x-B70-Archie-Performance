#!/usr/bin/env bash
# Prove speculative decoding is ACTIVE on the running endpoint, measured as a
# delta over bounded probe windows.
#
# Cumulative /metrics accept_len is not an attestation: it mixes the counting
# probe with thinking-heavy chat and dilutes toward ~1.6, so a broken engine can
# look fine and a fine engine can look broken depending on what ran before.
# Read acceptance as a bracketed counter delta around one probe type instead.
#
# accept_len is 1 + accepted/drafts, not accepted/drafts: the bonus token counts.
# An earlier version of this file omitted the 1 and its numbers were not
# comparable to the reference rows in the README.
#
# Reference windows for the deployed FP8 config (same formula, same settings):
#     C1-window   26 169 7.500
#     NAT-window  256 512 3.000
# At k=7 the ceiling is 8.000. Zero drafts is a hard failure -- spec decode
# silently disabling itself is exactly what this file exists to catch.
#
# Which of the three numbers to trust. The ACCEPTED count is the load-bearing one:
# across ~12 boots spanning bfloat16 and float16 it has read 512 every single time,
# with the C1 window pinned at 26 169 7.500. Draft counts are the soft ones -- the
# NAT window has also been read as 258 and as 272, and 272 drags accept_len from
# 3.000 to 2.882 (+16 drafts, +0 accepted). Treat that as a window-boundary
# artifact rather than a result: no mechanism for the +16 is established, so do not
# quote accept_len as evidence a config changed acceptance behaviour unless the
# accepted count moves with it.
#
# Why the boundary is a single read and not a poll. A drain() that waited for
# several consecutive equal readings was written and unit-tested as a stabiliser,
# then failed to remove the 272 in two independent runs and changed nothing on a
# healthy engine. Keeping it would have meant editing this file's read protocol AND
# the harness that pins it (falsify/acceptance-cases.sh scripts one /metrics
# response per read) on the strength of a causal story that did not hold up.
# Reverted for that reason; the note above is what survives of it.
set -uo pipefail

PORT="${PORT:-8103}"; MNAME="${SERVED_NAME:-secondary}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="${PROMPT_FILE:-$HERE/../../data/prompt-planning.txt}"

timeout 10 curl -sf -m 8 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "[acceptance] ABORT: /health not answering on $PORT"; exit 1; }

metrics(){ timeout 20 curl -s "http://127.0.0.1:$PORT/metrics" | python3 -c '
import sys,re
t=sys.stdin.read()
def g(n):
    m=re.search(r"^vllm:%s_total\{[^}]*\} ([0-9.e+]+)$"%n,t,re.M)
    return float(m.group(1)) if m else 0.0
print("%d %d" % (g("spec_decode_num_drafts"), g("spec_decode_num_accepted_tokens")))'; }

echo "[acceptance] windows on port $PORT (k=7 static, FULL_DECODE_ONLY)"
echo "[acceptance]   reference: C1 26 169 7.500 | NAT 256 512 3.000"

read -r D0 A0 <<<"$(metrics)"
c1=$(timeout 300 python3 "$HERE/probe.py" --url "http://127.0.0.1:$PORT/v1/completions" \
      --model "$MNAME" --prompt "1 2" --max-tokens 16 --iters 10 --warmup 3 2>&1 | tr -d '\n')
read -r D1 A1 <<<"$(metrics)"
nat=$(timeout 300 python3 "$HERE/probe.py" --url "http://127.0.0.1:$PORT/v1/completions" \
      --model "$MNAME" --prompt "$(cat "$PROMPT_FILE")" --max-tokens 96 --iters 6 --warmup 2 2>&1 | tr -d '\n')
read -r D2 A2 <<<"$(metrics)"

acc1=$(awk -v a=$D0 -v b=$A0 -v c=$D1 -v d=$A1 'BEGIN{D=c-a;A=d-b;printf "%d %d %s",D,A,(D>0?sprintf("%.3f",1+A/D):"n/a")}')
accn=$(awk -v a=$D1 -v b=$A1 -v c=$D2 -v d=$A2 'BEGIN{D=c-a;A=d-b;printf "%d %d %s",D,A,(D>0?sprintf("%.3f",1+A/D):"n/a")}')
echo "[acceptance]   C1-window  $acc1"
echo "[acceptance]   NAT-window $accn"
raw(){ echo "$1" | python3 -c 'import sys,json;t=sys.stdin.read();i=t.find("{");d=json.loads(t[i:]) if i>=0 else {};print("median_tok_s=%s peak=%s" % (d.get("median_tok_s_after_ttft"),d.get("peak_tok_s_after_ttft")))'; }
echo "[acceptance]   C1 raw  : $(raw "$c1")"
echo "[acceptance]   NAT raw : $(raw "$nat")"

ok=1
[[ $((D1-D0)) -gt 0 ]] || { echo "[acceptance]   FAIL: 0 drafts in the C1 window"; ok=0; }
[[ $((D2-D1)) -gt 0 ]] || { echo "[acceptance]   FAIL: 0 drafts in the NAT window"; ok=0; }

check_al(){ # $1=name $2=accept_len
  [[ "$2" == "n/a" ]] && return 0
  # NB: `awk 'BEGIN{exit !(x<1.5)}' || fire` is inverted -- awk exits 0 when the
  # condition is true, so `||` fires on a HEALTHY value and lets a broken one
  # through. Test positively.
  if awk -v x="$2" 'BEGIN{exit !(x<1.5)}'; then
    echo "[acceptance]   SUSPECT: $1 accept_len $2 -- drafts flowing but rejected en masse"
    return 1
  fi
  return 0
}
read -r _ _ AL1 <<<"$acc1"
read -r _ _ ALN <<<"$accn"
check_al C1 "$AL1" || ok=0
check_al NAT "$ALN" || ok=0

if [[ $ok -eq 1 ]]; then
  echo "[acceptance] PASS: drafts flowing with plausible acceptance in both windows"
else
  echo "[acceptance] FAIL: spec-decode attestation"
  exit 1
fi
