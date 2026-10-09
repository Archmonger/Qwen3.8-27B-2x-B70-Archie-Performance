#!/usr/bin/env bash
# Falsification harness for scripts/lib/attest.sh, the attestation verify.sh runs.
#
# Why this exists: an attestation whose verdict paths cannot fail is not an
# attestation. Two verdict paths in this project's history printed confident
# results from inputs that should have scored nothing -- a summary that reported
# "observed CORRUPT" from zero parsed rows, and a boot counter that counted lines
# instead of boots. So every rule attest.sh implements is driven here with a
# synthetic engine log and must produce the OPPOSITE verdict from the clean case.
#
# It sources scripts/lib/attest.sh rather than reimplementing the rules; a harness
# with its own copy of the logic keeps passing after the real check breaks.
#
# Usage:
#   scripts/falsify/verify-cases.sh                 # synthetic fixtures
#   CONTAINER=vllm-llmscaler scripts/falsify/verify-cases.sh
#                                                 # + cross-check the fixture against
#                                                   # the real engine log (recommended)
# Exit 0 only if every case discriminates, the clean baseline is clean, and the
# labels cover every rule.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/../lib/attest.sh"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

WANT_CEIL=80
WANT_K=7
WANT_KV=fp8

# The fixture is a Python-dict config dump in the shape vLLM actually prints: keys
# carrying a trailing quote, `<CUDAGraphMode.X: (n, n)>`, a closing brace on the last
# key. attest.sh's extraction regexes are written against that repr, so a prettified
# fixture would exercise patterns that never match in production -- which is exactly
# how an attestation ends up checking nothing and reporting success. This fixture was
# transcribed from a real engine log; the CONTAINER= cross-check below is what keeps
# it honest. If you hand-edit this fixture, run that check.
write_good() {
  cat > "$1" <<'EOF'
INFO CompilationConfig(mode='none', cudagraph_mode': <CUDAGraphMode.FULL_DECODE_ONLY: (2, 2)>, max_cudagraph_capture_size': 80, cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80])
INFO SpeculativeConfig(method': 'dspark', model': '/models/drafter', num_speculative_tokens': 7}
INFO CacheConfig(cache_dtype': 'fp8')
INFO Using V2 Model Runner
INFO Graph capturing finished in 6 secs, took 0.58 GiB
EOF
}

verdict_of() { # $1=log $2=rule-name -> ok|bad|<empty>
  attest "$1" "$WANT_CEIL" "$WANT_K" "$WANT_KV" | awk -F'|' -v n="$2" '$2==n{print $3}'
}

fails=0

# --- baseline: the clean log must be entirely clean. Without this, a case that
#     appears to discriminate because the baseline itself is broken means nothing.
write_good "$WORK/good.log"
badlines=$(attest "$WORK/good.log" "$WANT_CEIL" "$WANT_K" "$WANT_KV" | awk -F'|' '$3=="bad"')
if [[ -n "$badlines" ]]; then
  echo "FAIL baseline: the clean fixture is not clean, so no case below means anything"
  echo "$badlines" | sed 's/^/       /'
  echo "       (if vLLM changed its config repr, fix write_good and rerun)"
  exit 1
fi
echo "ok   baseline: clean fixture yields no bad verdicts"

# --- empty log: every rule must report bad. Refusing to score an aggregate over
#     zero rows is the guard an earlier harness of this project lacked.
: > "$WORK/empty.log"
if attest "$WORK/empty.log" "$WANT_CEIL" "$WANT_K" "$WANT_KV" | grep -q '|ok|'; then
  echo "FAIL empty log produced an 'ok' verdict:"
  attest "$WORK/empty.log" "$WANT_CEIL" "$WANT_K" "$WANT_KV" | grep '|ok|' | sed 's/^/       /'
  fails=$((fails+1))
else
  echo "ok   empty-log case: every rule reports bad"
fi

# --- one case per rule: name | anchor | replacement | rule-that-must-flip.
# A case where the rule stays ok is a verdict path that cannot fail; a case whose
# anchor is absent from the fixture is reported broken rather than passing.
REMOVED='<<LINE-REMOVED>>'
cases=(
  "ceiling-72          |max_cudagraph_capture_size': 80|max_cudagraph_capture_size': 72|ceil"
  "ladder-truncated    |64, 72, 80]|64, 72]|ladder"
  "graphs-off          |<CUDAGraphMode.FULL_DECODE_ONLY: (2, 2)>|<CUDAGraphMode.NONE: (4, 4)>|cg"
  "k-4                 |num_speculative_tokens': 7}|num_speculative_tokens': 4}|k"
  "method-dflash       |method': 'dspark'|method': 'dflash'|meth"
  "schedule-active     |num_speculative_tokens': 7}|num_speculative_tokens': 7, num_speculative_tokens_per_batch_size': [[1, 1, 7], [2, 64, 4]]}|sched"
  "no-capture          |Graph capturing finished in 6 secs, took 0.58 GiB|$REMOVED|capture"
  "traceback           |Using V2 Model Runner|Traceback (most recent call last):|fatal"
  "v1-runner           |Using V2 Model Runner|Using V1 Model Runner|v2"
  "kv-auto             |cache_dtype': 'fp8'|cache_dtype': 'auto'|kv"
)

for c in "${cases[@]}"; do
  IFS='|' read -r name anchor repl rule <<<"$c"
  # Trim the padding used for alignment.
  name="$(echo "$name" | xargs)"; rule="$(echo "$rule" | xargs)"
  write_good "$WORK/c.log"
  # python for the substitution: these anchors are full of sed metacharacters
  # ([ ] : *), and a silent no-op mutation would make a case pass for the wrong reason.
  if ! A="$anchor" R="$repl" F="$WORK/c.log" python3 - <<'PY'
import os, sys
p = os.environ["F"]
s = open(p).read()
a, r = os.environ["A"], os.environ["R"]
if a not in s:
    sys.exit("anchor absent from the fixture: %r" % a)
open(p, "w").write(s.replace(a, r, 1))
PY
  then
    echo "FAIL $name: fixture lacks the anchor -- the case is broken, not passing"
    fails=$((fails+1)); continue
  fi
  # Guard the guard: a mutation that did not land proves nothing.
  if [[ "$repl" != "$REMOVED" ]] && ! grep -qF -- "${repl:0:20}" "$WORK/c.log"; then
    echo "FAIL $name: mutation absent from the mutated log -- vacuous case"
    fails=$((fails+1)); continue
  fi
  v=$(verdict_of "$WORK/c.log" "$rule")
  if [[ "$v" == ok || -z "$v" ]]; then
    echo "FAIL $name: rule '$rule' verdict is '${v:-none}' -- this verdict path cannot fail"
    fails=$((fails+1))
  else
    echo "ok   $name -> $rule=$v"
  fi
done

# --- labels must cover exactly the rules attest emits, or verify.sh silently drops
#     a verdict (or invents one).
rules_emitted=$(attest "$WORK/good.log" "$WANT_CEIL" "$WANT_K" "$WANT_KV" | awk -F'|' '{print $2}' | sort -u)
rules_labelled=$(attest_labels "$WANT_CEIL" "$WANT_K" "$WANT_KV" | awk -F'|' '{print $2}' | sort -u)
if [[ "$rules_emitted" != "$rules_labelled" ]]; then
  echo "FAIL attest and attest_labels disagree:"
  diff <(echo "$rules_emitted") <(echo "$rules_labelled") | sed 's/^/       /'
  fails=$((fails+1))
else
  echo "ok   rule/label coverage: $(echo "$rules_emitted" | wc -l | xargs) rules, all labelled"
fi

# --- optional cross-check: the fixture must extract the SAME values a real engine
#     log does. Catches vLLM repr drift that would otherwise turn every rule into a
#     no-op ("key not found" and "key wrong" both read as bad, so a broken regex on a
#     real log looks exactly like a correctly failing check).
if [[ -n "${CONTAINER:-}" ]] && command -v docker >/dev/null 2>&1 \
   && docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  docker logs "$CONTAINER" > "$WORK/live.log" 2>&1
  # Test the file, not a second `docker logs | grep -q .`: grep -q exits after one
  # line, docker logs takes SIGPIPE, and under `set -o pipefail` the pipeline reports
  # failure -- so a container with plenty of logs read as "no logs" and the check
  # silently skipped.
  if [[ ! -s "$WORK/live.log" ]]; then
    echo "SKIP live cross-check: no logs for $CONTAINER"
  else
    attest "$WORK/live.log" "$WANT_CEIL" "$WANT_K" "$WANT_KV" > "$WORK/live.verdicts"
    if awk -F'|' '$3=="bad"' "$WORK/live.verdicts" | grep -q .; then
      echo "NOTE live cross-check: the real engine disagrees with the fixture expectations"
      awk -F'|' '$3=="bad"{print "       "$2": "$4}' "$WORK/live.verdicts"
      echo "       If the engine is NOT the measured config, that is the check working."
      echo "       If it IS, then a rule is misextracting -- investigate before shipping."
      fails=$((fails+1))
    else
      echo "ok   live cross-check: $CONTAINER passes every rule the fixture passes"
    fi
  fi
else
  echo "note live cross-check skipped (set CONTAINER=<name> on the serving host to enable)"
fi

echo
if [[ $fails -eq 0 ]]; then
  echo "falsify/verify-cases: PASS -- ${#cases[@]} rules discriminated, baseline clean, labels complete"
  exit 0
fi
echo "falsify/verify-cases: FAIL ($fails problem(s)) -- do not trust the attestation until this is green"
exit 1
