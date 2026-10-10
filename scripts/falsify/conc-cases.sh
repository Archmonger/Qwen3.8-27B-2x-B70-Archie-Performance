#!/usr/bin/env bash
# Falsification harness for the conc.py --max-tokens canonical gate.
#
# Why this guard needs proving at all: the gate exists because an unrecorded
# --max-tokens default once made a provably-correct engine look ~13% slower at
# high concurrency, and that was diagnosed as a host fault. A guard that does not
# actually reject is worse than no guard, because it launders the mistake into a
# "we checked" state.
#
# The guard must fail BOTH ways, so both directions are asserted:
#   * it must fire on a non-canonical value, and
#   * it must NOT fire on the canonical value.
#
# How both directions are proven without a live engine: the guard returns rc=3
# before it builds a URL or opens a socket, whereas a run that walks past the guard
# reaches the network and comes back rc=2 (WARMUP_FAILED, connection refused).
# Against a dead port those two codes are therefore unambiguous, and they are
# distinct codes on purpose -- the warmup path already used rc=2, so the guard
# could not share it (same reason soak.sh keeps 124 and 143 apart). Nothing here
# touches a serving endpoint, so this harness runs anywhere.
#
# See docs/verification.md for the general discipline.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONC="$HERE/../bench/conc.py"
DEAD_PORT="${DEAD_PORT:-1}"   # port 1 is not a vLLM; connection refused, fast

[[ -f "$CONC" ]] || { echo "conc-cases: missing $CONC" >&2; exit 2; }

fails=0
ok()  { echo "  ok   $*"; }
bad() { echo "  FAIL $*"; fails=$((fails+1)); }

# $1=description $2=expected-rc $3...=conc.py args
expect_rc() {
  local desc="$1" want="$2"; shift 2
  local out rc
  out=$(python3 "$CONC" --port "$DEAD_PORT" --model nonexistent "$@" 2>&1); rc=$?
  if [[ "$rc" == "$want" ]]; then
    ok "$desc (rc=$rc)"
  else
    bad "$desc: expected rc=$want, got $rc"
    printf '%s\n' "$out" | sed 's/^/         | /'
  fi
  LAST_OUT="$out"
}

echo "=== conc.py --max-tokens guard falsification ==="

# 1. The guard must fire: non-canonical value, no acknowledgement.
expect_rc "max_tokens=96 (the historic trap) rejected" 3 --cells 10:30 --max-tokens 96
if grep -q 'confirm-noncanonical' <<<"${LAST_OUT:-}"; then
  ok "rejection names the escape hatch"
else
  bad "rejection does not tell the user how to proceed"
fi
if grep -qE 'not the canonical 256' <<<"${LAST_OUT:-}"; then
  ok "rejection states the canonical value"
else
  bad "rejection does not state the canonical value"
fi

# A second, different non-canonical value must also be caught -- otherwise the
# guard may merely be hard-coded to reject 96 and the whole class is unguarded.
expect_rc "max_tokens=160 rejected" 3 --cells 10:30 --max-tokens 160

# 2. The guard must NOT fire on the canonical value. rc=2 (not 3) proves it got
#    past the guard and only then failed at the dead port.
expect_rc "canonical (no --max-tokens) passes the guard" 2 --cells 10:30
expect_rc "explicit --max-tokens 256 passes the guard" 2 --cells 10:30 --max-tokens 256

# 3. The acknowledgement must be honoured.
expect_rc "non-canonical + --confirm-noncanonical passes" 2 \
  --cells 10:30 --max-tokens 96 --confirm-noncanonical

# 4. No side effects before the guard fires: a rejected run must print no cell
#    data on stdout, or it can be mistaken for a measurement.
out=$(python3 "$CONC" --port "$DEAD_PORT" --model x --cells 10:30 --max-tokens 96 2>/dev/null)
if [[ -z "${out// /}" ]]; then
  ok "rejected run prints nothing on stdout (cannot be mistaken for a result)"
else
  bad "rejected run printed stdout: $out"
fi

# 5. The escape hatch must be discoverable, or the guard just looks broken.
if python3 "$CONC" --help 2>&1 | grep -q -- '--confirm-noncanonical'; then
  ok "--confirm-noncanonical is discoverable in --help"
else
  bad "--confirm-noncanonical missing from --help"
fi

# ---------------------------------------------------------------------------
# 6. The per-test budget guard.
#
# Each named test in conc.py's TESTS table pins its own --max-tokens, so the gate has
# to resolve per test rather than against one global constant. Both directions are
# asserted per test: a run at the test's own pinned value must pass, and a run at
# something else must be rejected. One direction alone would not catch the table being
# read from the wrong entry -- the value would still be a real pinned value and the
# guard would still be self-consistent, while the row quietly measured another budget.
#
# The pinned values are read out of the table by importing it, not duplicated here, so
# this cannot drift into a second copy of the truth that passes while conc.py changes.
# They are ALSO compared against EXPECTED below: importing alone would happily agree
# with a table whose numbers were mistyped, and these budgets are what the README's
# Max Tokens column quotes.
# Alphabetical, because the table above is built from sorted(TESTS.items()). Comparing
# against a hand-ordered literal would make this fail on a rename that changed nothing
# but sort position.
EXPECTED="coding=2048 counting=512 planning=2048 prose=2048"

# The extraction is a heredoc, not `python3 -c '...\"key\"...'`. Inside single quotes
# bash keeps the backslashes, so Python receives \" and dies with "unexpected character
# after line continuation character" -- which, with stderr swallowed and the result read
# as an empty string, looked exactly like a table of no entries. Empty must never be a
# pass, so the exit code is checked explicitly and this whole block is a no-go without it.
table=$(cd "$HERE/../bench" && python3 - <<'PY' 2>&1
import conc
print(" ".join(f"{k}={v['max_tokens']}" for k, v in sorted(conc.TESTS.items())))
PY
)
table_rc=$?
if [[ $table_rc -ne 0 ]]; then
  bad "cannot import conc.TESTS (rc=$table_rc): $table"
elif [[ -z "$table" ]]; then
  bad "conc.TESTS imported but yielded no entries -- an empty table is not a pass"
else
  if [[ "$table" == $EXPECTED ]]; then
    ok "pinned budgets match the recorded values ($table)"
  else
    bad "pinned budgets drifted: table=[$table] expected=[$EXPECTED]"
  fi

  for kv in $table; do
    t="${kv%%=*}"; mt="${kv##*=}"
    expect_rc "--test $t at its pinned $mt passes the guard" 2 \
      --test "$t" --cells 10:30
    expect_rc "--test $t at 96 rejected" 3 --test "$t" --cells 10:30 --max-tokens 96
    expect_rc "--test $t at 96 with --confirm-noncanonical passes" 2 \
      --test "$t" --cells 10:30 --max-tokens 96 --confirm-noncanonical
  done

  # 7. Every test must name a prompt that actually exists. A missing file raises at
  #    run time, but only after the guard and after the engine has been warm -- so it
  #    is cheaper and clearer to catch it here, and a typo'd name would otherwise look
  #    like an engine fault.
  #    Heredoc for the same reason as the table above, and rc is checked: this check's
  #    pass condition is *empty output*, so a Python syntax error used to satisfy it.
  missing=$(cd "$HERE/../bench" && python3 - <<'PY' 2>&1
import os, conc
bad = [f"{k}:{v['prompt']}" for k, v in sorted(conc.TESTS.items())
       if not os.path.isfile(os.path.join(conc._DATA, v["prompt"]))]
print(" ".join(bad))
PY
)
  missing_rc=$?
  if [[ $missing_rc -ne 0 ]]; then
    bad "prompt-existence check could not run (rc=$missing_rc): $missing"
  elif [[ -z "$missing" ]]; then
    ok "every test names a prompt that exists in data/"
  else
    bad "tests name missing prompt files: $missing"
  fi
fi

# 8. --test must be discoverable, or the table is invisible to whoever runs this.
if python3 "$CONC" --help 2>&1 | grep -q -- '--test'; then
  ok "--test is discoverable in --help"
else
  bad "--test missing from --help"
fi

echo
if [[ $fails -eq 0 ]]; then
  echo "falsify/conc-cases: PASS -- guard fires on non-canonical, leaves canonical alone"
  exit 0
fi
echo "falsify/conc-cases: FAIL ($fails)"
exit 1
