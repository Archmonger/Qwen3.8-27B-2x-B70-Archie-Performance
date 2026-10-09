#!/usr/bin/env bash
# Falsification harness for scripts/bench/acceptance.sh's two guards.
#
#   guard 1  zero drafts in a probe window must be a hard FAIL. Spec decode can
#            disable itself silently, and a dead spec engine otherwise records
#            "0 0 n/a" -- which reads like a valid measurement of no drafts.
#   guard 2  accept_len near 1.0 with drafts flowing must be flagged SUSPECT rather
#            than passed. That is the shape of "output correct, graph feeding
#            garbage": verification catches the bad tokens, so nothing but this guard
#            distinguishes it from a healthy engine.
#
# /metrics is served from a scripted response sequence so each window delta is
# exactly what the case specifies. The probe traffic is allowed to fail; the guards
# consume only the counters, and the guards are what is under test.
#
# Usage:  scripts/falsify/acceptance-cases.sh
# Exit 0 only if the accept_len formula is right and both guards can fire.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$HERE/../bench"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
fails=0

# --- 1. the arithmetic, the part most able to invert silently -------------------
# accept_len is 1 + accepted/drafts, because the bonus token counts. An earlier
# version of acceptance.sh omitted the 1, so its numbers were not comparable to the
# reference rows printed alongside them.
AL=$(awk -v a=0 -v b=0 -v c=26 -v d=169 \
      'BEGIN{D=c-a;A=d-b;printf "%s",(D>0?sprintf("%.3f",1+A/D):"n/a")}')
if [[ "$AL" == "7.500" ]]; then
  echo "ok   accept_len formula -> $AL (reference: C1 26 drafts / 169 accepted = 7.500)"
else
  echo "FAIL accept_len gave '$AL', expected 7.500 -- the formula drifted from the reference"
  fails=$((fails+1))
fi
# Tested in both directions. `awk 'BEGIN{exit !(x<1.5)}' || fire` is inverted: awk
# exits 0 when the condition holds, so `||` fires on a HEALTHY value and lets a broken
# one through. acceptance.sh tests positively; this proves the predicate does that.
if awk -v x="1.000" 'BEGIN{exit !(x<1.5)}'; then
  echo "ok   guard predicate fires on accept_len 1.000"
else
  echo "FAIL guard predicate does NOT fire on 1.000 -- the test is inverted"
  fails=$((fails+1))
fi
if awk -v x="7.500" 'BEGIN{exit !(x<1.5)}'; then
  echo "FAIL guard predicate fires on a HEALTHY 7.500"
  fails=$((fails+1))
else
  echo "ok   guard predicate leaves 7.500 alone"
fi

# --- 2. drive the real acceptance.sh against scripted /metrics ------------------
metrics_line() { # $1=drafts $2=accepted
  printf '# HELP x\nvllm:spec_decode_num_drafts_total{model_name="secondary"} %s\nvllm:spec_decode_num_accepted_tokens_total{model_name="secondary"} %s\n' "$1" "$2"
}

# run_case <name> <expect: fail|flag|pass> <response>...
# Responses are returned one per /metrics read; the final one repeats.
run_case() {
  local name="$1" want="$2"; shift 2
  local dir; dir="$(mktemp -d "$WORK/case.XXXX")"
  mkdir -p "$dir/bin" "$dir/resp"
  local n=0 resp
  for resp in "$@"; do printf '%s\n' "$resp" > "$dir/resp/$n"; n=$((n+1)); done
  local last=$((n-1))
  cat > "$dir/bin/curl" <<EOF
#!/bin/bash
args="\$*"
if [[ "\$args" == *"/health"* ]]; then exit 0; fi
if [[ "\$args" == *"/metrics"* ]]; then
  n=\$(cat "$dir/idx" 2>/dev/null || echo 0)
  [[ \$n -gt $last ]] && n=$last
  cat "$dir/resp/\$n"
  echo \$((n+1)) > "$dir/idx"
  exit 0
fi
exec /usr/bin/curl "\$@"
EOF
  chmod +x "$dir/bin/curl"

  local out rc
  out="$(PATH="$dir/bin:$PATH" PORT=8103 SERVED_NAME=secondary timeout 600 \
         bash "$BENCH/acceptance.sh" 2>&1)"; rc=$?
  if grep -q "ABORT: /health" <<<"$out"; then
    echo "FAIL $name: the stub /health was rejected -- the case is broken, not passing"
    fails=$((fails+1)); return
  fi
  case "$want" in
    fail)
      if [[ $rc -ne 0 ]] && grep -q "0 drafts" <<<"$out"; then
        echo "ok   $name: zero-draft windows rejected (rc=$rc)"
      else
        echo "FAIL $name: rc=$rc -- zero drafts did not fail the attestation"
        grep -E "C1-window|NAT-window" <<<"$out" | sed 's/^/       /'
        fails=$((fails+1))
      fi ;;
    flag)
      if grep -q "SUSPECT" <<<"$out"; then
        echo "ok   $name: drafts with zero accepted flagged SUSPECT"
      else
        echo "FAIL $name: an engine accepting nothing passed the attestation"
        grep -E "C1-window|NAT-window" <<<"$out" | sed 's/^/       /'
        fails=$((fails+1))
      fi ;;
    pass)
      if [[ $rc -eq 0 ]] && ! grep -q "SUSPECT" <<<"$out"; then
        echo "ok   $name: healthy counters pass both guards"
      else
        echo "FAIL $name: the healthy control did not pass, so nothing below it means anything"
        grep -E "C1-window|NAT-window|SUSPECT" <<<"$out" | sed 's/^/       /'
        fails=$((fails+1))
      fi ;;
  esac
}

echo
echo "=== acceptance.sh guard falsification ==="

# Control first: if the healthy case cannot pass, the failures below prove nothing.
# acceptance.sh reads /metrics three times (before C1, between, after NAT), so the
# first three responses are what the windows are computed from.
run_case "control: healthy deltas      " pass \
  "$(metrics_line 100 100)" "$(metrics_line 126 269)" "$(metrics_line 152 421)" \
  "$(metrics_line 152 421)"

# Guard 1: counters never move.
run_case "counters frozen (0 drafts)   " fail \
  "$(metrics_line 100 700)" "$(metrics_line 100 700)" "$(metrics_line 100 700)"

# Guard 2: drafts advance in both windows, accepted never does -> accept_len 1.000.
run_case "drafts, nothing accepted     " flag \
  "$(metrics_line 100 100)" "$(metrics_line 126 100)" "$(metrics_line 152 100)"

# Guard 1 again with only the second window dead: one dead window is still a fail.
run_case "only the NAT window is dead  " fail \
  "$(metrics_line 100 100)" "$(metrics_line 126 269)" "$(metrics_line 126 269)"

echo
if [[ $fails -eq 0 ]]; then
  echo "falsify/acceptance-cases: PASS -- formula matches the reference and both guards can fire"
  exit 0
fi
echo "falsify/acceptance-cases: FAIL ($fails problem(s)) -- do not trust acceptance.sh"
exit 1
