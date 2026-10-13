#!/usr/bin/env bash
# Degeneracy check for measured arms, shared by scripts/bench/arm.sh and
# scripts/falsify/arm-cases.sh.
#
# Usage:  source scripts/lib/degenerate.sh
#         assert_not_degenerate <conc.json> <c1raw.json> <natraw.json>
#           -> prints "DEGENERATE: <reasons>" on stdout and returns 1 when the
#              measurements did not actually happen; returns 0 when they did.
#
# Why this exists. One fp16 boot recorded gate=PASS with conc5 and conc1 at 0
# tokens / 0.0 s, conc10 at 15.13 tok/s while the same config measured 237-240 on
# its other two boots, and both single-stream probes unparseable. Nothing caught
# it: the only structural test asked whether the string "conc10" appeared, so a
# JSON skeleton of zeros passed. A stalled or half-dead boot therefore became a
# publishable row that advertised itself as a passing arm.
#
# The rules, and why each is the one that matters:
#   * tokens is the load-bearing field. aggregate_tok_s of 0.0 alongside 0 tokens
#     means no request completed; a genuinely slow engine still moves tokens.
#     Testing tok/s alone would reject the real thing and accept a cell whose
#     requests all errored.
#   * every cell is checked, not just the cap. A run can complete its first cell
#     and then have the engine stall, which is what happened.
#   * probe output must PARSE, not merely be non-empty. probe.py prints JSON on
#     stdout; a crash leaves "null", and `null` is a missing number, not a low one.
#   * the median must be positive, because a probe that times out every iteration
#     still prints well-formed JSON with a null/0 median.
#
# Extracted rather than inlined for the same reason attest.sh was: a harness that
# re-implements the rule drifts from it, and a drifting harness keeps passing while
# the real check is broken.

assert_not_degenerate(){
  local conc_f=${1:?conc json file} c1_f=${2:?C1 raw file} nat_f=${3:?NAT raw file}
  # Where does jsonobj.py live? Derive it from THIS file's own path, not the
  # caller's cwd, so the gate works when sourced from arm.sh, from a falsify
  # harness, or from an interactive shell. DG_BENCH_DIR overrides it: a MUTANT
  # copy of this file sits in a temp directory with no sibling bench/ folder, so
  # file-relative resolution cannot work there and the harness names the directory
  # explicitly instead. (Left to itself the mutant fails the import, python exits
  # nonzero, and every mutation reads as "still rejected" -- which the harness would
  # report as "this rule is not load-bearing", a verdict caused entirely by a broken
  # import path rather than by the mutation. Wrong-for-the-reason green and
  # wrong-for-the-reason red are equally bad.)
  local dg_dir
  dg_dir=${DG_BENCH_DIR:-}
  if [[ -n "$dg_dir" ]]; then
    dg_dir=$(cd -- "$dg_dir" 2>/dev/null && pwd) || dg_dir=
  else
    dg_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../bench" 2>/dev/null && pwd) || dg_dir=
  fi
  if [[ -z "$dg_dir" ]]; then
    echo "DEGENERATE: cannot locate scripts/bench (DG_BENCH_DIR='$DG_BENCH_DIR')"
    return 1
  fi
  A_CONC_F="$conc_f" A_C1RAW_F="$c1_f" A_NATRAW_F="$nat_f" \
  PYTHONPATH="$dg_dir${PYTHONPATH:+:$PYTHONPATH}" python3 -c "
import json, os, sys
try:
    from jsonobj import jsonobj
except ImportError as exc:
    # Fail LOUD. A silent import failure would exit nonzero with an empty stdout,
    # which upstream reads as "degenerate" with a blank reason: the arm is rejected
    # for a plumbing fault nobody can see, and in a mutation harness it masquerades
    # as evidence that a real rule is not load-bearing.
    print('DEGENERATE: cannot import jsonobj from PYTHONPATH (%s)' % exc)
    sys.exit(1)

def load(p):
    # The SAME parser arm.sh uses to extract its numbers, imported rather than
    # re-typed: if the extractor and the rejection gate ever disagree about what
    # parses, a measurement is either recorded unchecked or thrown away.
    return jsonobj(p)

bad = []
c = load(os.environ['A_CONC_F'])
cells = c.get('cells') if isinstance(c, dict) else None
if not isinstance(cells, dict):
    bad.append('conc json unreadable')
else:
    for name in ('conc10', 'conc5', 'conc1'):
        cell = cells.get(name)
        if not isinstance(cell, dict):
            bad.append('%s missing' % name)
            continue
        if not cell.get('tokens'):
            bad.append('%s completed 0 requests' % name)
        elif not (cell.get('aggregate_tok_s') or 0) > 0:
            bad.append('%s reported 0 tok/s while tokens were present' % name)

for tag, key, need in (('C1', 'A_C1RAW_F', 'A_C1_NEED'), ('NAT', 'A_NATRAW_F', 'A_NAT_NEED')):
    p = os.environ[key]
    want = int(os.environ.get(need) or 0)
    try:
        raw = open(p).read().strip()
    except OSError:
        bad.append('%s probe output missing' % tag)
        continue
    if raw in ('', 'null'):
        bad.append('%s probe did not parse' % tag)
        continue
    d = load(p) or {}
    if not (d.get('median_tok_s_after_ttft') or 0) > 0:
        bad.append('%s probe has no positive median' % tag)
        continue
    # A median over one surviving sample is not a measurement. probe.py catches
    # every per-iteration exception and continues, so a boot where 9 of 10
    # iterations errored still prints well-formed JSON with a healthy-looking
    # median, and "samples" counts only the successes. Checking the median alone
    # accepts that. Requiring the iteration count to have been met is what makes a
    # partially-dead engine a rejected arm rather than a fast one.
    # (No backticks in this comment: it sits inside a double-quoted python -c
    # string, so bash would run them as command substitution and corrupt the code.)
    if want and (d.get('samples') or 0) < want:
        bad.append('%s probe completed %s/%s iterations'
                   % (tag, d.get('samples'), want))

if bad:
    print('DEGENERATE: ' + '; '.join(bad))
    sys.exit(1)
"
}
