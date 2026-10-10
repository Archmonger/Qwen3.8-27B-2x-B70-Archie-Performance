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

# --- boot_slice: the slice that decides WHICH boot the fatal rule gets to see.
#     This is safety-relevant in both directions: a slice that keeps too little hides a
#     real crash in the current boot, and a slice that keeps everything re-inherits
#     crashes from long-dead boots (the bug that made attestation permanently red after
#     one DEVICE_LOST). Each case therefore asserts membership, not just line counts.
echo
echo "=== boot_slice ==="
{
  echo '2026-10-09T02:59:50.000000000Z (APIServer pid=418) INFO starting boot one'
  echo '2026-10-09T21:25:56.000000000Z (Worker_TP1) RuntimeError: UR_RESULT_ERROR_DEVICE_LOST'
  echo '2026-10-09T21:26:06.000000000Z (APIServer pid=419) INFO starting boot two'
  echo '2026-10-10T03:44:51.000000000Z (APIServer pid=419) INFO Engine 000: Running: 0 reqs'
} > "$WORK/two-boots.log"

check_slice() { # $1=desc $2=want-rc $3=since $4=must-contain $5=must-not-contain
  local desc="$1" want_rc="$2" since="$3" must="$4" mustnot="$5" rc=0 kept
  # boot_slice prints the kept-line count; capture it so the harness output stays one
  # line per case. The count is asserted through membership below, not through this.
  kept=$(boot_slice "$WORK/two-boots.log" "$since" "$WORK/sliced.log") && rc=0 || rc=$?
  if [[ "$rc" != "$want_rc" ]]; then
    echo "FAIL $desc: rc=$rc expected $want_rc"; fails=$((fails+1)); return
  fi
  if [[ -n "$must" ]] && ! grep -qF -- "$must" "$WORK/sliced.log"; then
    echo "FAIL $desc: expected line '$must' is missing from the slice"; fails=$((fails+1)); return
  fi
  if [[ -n "$mustnot" ]] && grep -qF -- "$mustnot" "$WORK/sliced.log"; then
    echo "FAIL $desc: dead-boot line '$mustnot' LEAKED into the slice"; fails=$((fails+1)); return
  fi
  echo "ok   $desc"
}

# Boot two is the live one: it must contain its own lines and must not carry the dead
# boot's DEVICE_LOST, which is the whole point of slicing.
check_slice "slices to the current boot" 0 '2026-10-09T21:26:06.222976221Z' \
  'starting boot two' 'UR_RESULT_ERROR_DEVICE_LOST'
# Same-second start: StartedAt carries nanoseconds, so a second-granularity comparison
# must still keep the first line of the boot or the banner/capture lines get dropped.
check_slice "keeps lines from the starting second" 0 '2026-10-09T21:26:06.000000000Z' \
  'starting boot two' 'boot one'
# FAIL-CLOSED: no usable boot start must fall back to the whole log, never to a narrower
# window. A slice that quietly narrowed on parse failure would turn a missing timestamp
# into a pass on the fatal rule.
check_slice "missing boot start falls back to the full log" 0 '' \
  'UR_RESULT_ERROR_DEVICE_LOST' ''
check_slice "garbage boot start falls back to the full log" 0 'n/a' \
  'UR_RESULT_ERROR_DEVICE_LOST' ''
# An empty source is not "a clean log": it must fail rather than attest nothing.
: > "$WORK/empty.log"
if kept=$(boot_slice "$WORK/empty.log" '2026-10-09T21:26:06.222976221Z' "$WORK/s0.log"); then
  echo "FAIL empty source log was accepted as a clean boot"; fails=$((fails+1))
else
  echo "ok   empty source log is rejected, not attested"
fi
# A boot start after every log line yields an empty slice, which must be an error.
if kept=$(boot_slice "$WORK/two-boots.log" '2027-01-01T00:00:00.000000000Z' "$WORK/s1.log"); then
  echo "FAIL a boot start after all logs was accepted as clean"; fails=$((fails+1))
else
  echo "ok   future boot start is rejected, not silently clean"
fi

# --- the fatal rule must actually consume the slice, and this must be proved in BOTH
#     directions. Without this, boot_slice could be correct and unused -- how a fix
#     becomes decoration -- or could silently widen back to the whole log and still
#     print a green fatal verdict.
#
#     Both cases use the CLEAN config fixture plus a crash line, so every other rule has
#     something to match. An earlier version ran fatal against a slice of two-boots.log,
#     which has no config lines at all: fatal there was 'ok' for the wrong reason
#     (nothing to extract), which is a pass that proves nothing.
write_good "$WORK/clean.cfg"
{ cat "$WORK/clean.cfg"
  echo '2026-10-09T21:25:56.000000000Z (Worker_TP1) RuntimeError: UR_RESULT_ERROR_DEVICE_LOST'
  echo '2026-10-09T21:26:06.000000000Z (APIServer pid=419) INFO starting boot two'
  echo '2026-10-09T21:31:00.000000000Z (Worker_TP1) RuntimeError: UR_RESULT_ERROR_DEVICE_LOST'
} > "$WORK/two-crashes.log"

# (a) crash inside the boot -> fatal MUST be bad. Proves the slice is read at all.
if boot_slice "$WORK/two-crashes.log" '2026-10-09T21:26:06.222976221Z' "$WORK/bad-boot.log" \
   >/dev/null; then
  fv=$(verdict_of "$WORK/bad-boot.log" fatal)
  if [[ "$fv" == bad ]]; then
    echo "ok   fatal fires on a crash inside the boot window"
  else
    echo "FAIL fatal verdict is '$fv' with a crash INSIDE the boot -- rule is not reading the slice"
    fails=$((fails+1))
  fi
else
  echo "FAIL could not slice the two-crash fixture"; fails=$((fails+1))
fi

# (b) crash only in the dead boot -> fatal MUST be ok. Proves the boundary is honoured,
#     i.e. the slice is not quietly the whole log.
if boot_slice "$WORK/two-crashes.log" '2026-10-09T21:26:06.222976221Z' "$WORK/good-boot.log" \
   >/dev/null; then
  # Re-slice from a log whose ONLY crash predates the boot start.
  { cat "$WORK/clean.cfg"
    echo '2026-10-09T21:25:56.000000000Z (Worker_TP1) RuntimeError: UR_RESULT_ERROR_DEVICE_LOST'
    echo '2026-10-09T21:26:06.000000000Z (APIServer pid=419) INFO starting boot two'
  } > "$WORK/one-crash.log"
  boot_slice "$WORK/one-crash.log" '2026-10-09T21:26:06.222976221Z' "$WORK/good-boot.log" \
    >/dev/null
  fv=$(verdict_of "$WORK/good-boot.log" fatal)
  if [[ "$fv" == ok ]]; then
    echo "ok   fatal clears a crash that lives only in the dead boot"
  else
    echo "FAIL fatal verdict is '$fv' with no crash in this boot -- the slice is wider than the boot"
    fails=$((fails+1))
  fi
else
  echo "FAIL could not slice the one-crash fixture"; fails=$((fails+1))
fi

# --- optional cross-check: the fixture must extract the SAME values a real engine
#     log does. Catches vLLM repr drift that would otherwise turn every rule into a
#     no-op ("key not found" and "key wrong" both read as bad, so a broken regex on a
#     real log looks exactly like a correctly failing check).
if [[ -n "${CONTAINER:-}" ]] && command -v docker >/dev/null 2>&1 \
   && docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  # Timestamps + boot_slice, matching verify.sh exactly. Cross-checking the whole log
  # while the live check reads one boot would let the two disagree about which lines are
  # even in scope, which is the drift this file exists to prevent.
  docker logs --timestamps "$CONTAINER" > "$WORK/live.raw" 2>&1
  live_started=$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER" 2>/dev/null)
  if ! boot_slice "$WORK/live.raw" "$live_started" "$WORK/live.log"; then
    echo "  WARN live boot slice empty; cross-checking the full log"
    cat "$WORK/live.raw" > "$WORK/live.log"
  fi
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
