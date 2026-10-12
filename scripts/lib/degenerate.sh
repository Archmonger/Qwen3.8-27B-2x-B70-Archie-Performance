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
  A_CONC_F="$conc_f" A_C1RAW_F="$c1_f" A_NATRAW_F="$nat_f" python3 -c "
import json, os, sys

def load(p):
    try:
        return json.loads(open(p).read().strip())
    except Exception:
        return None

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

for tag, key in (('C1', 'A_C1RAW_F'), ('NAT', 'A_NATRAW_F')):
    p = os.environ[key]
    try:
        raw = open(p).read().strip()
    except OSError:
        bad.append('%s probe output missing' % tag)
        continue
    if raw in ('', 'null'):
        bad.append('%s probe did not parse' % tag)
        continue
    if not ((load(p) or {}).get('median_tok_s_after_ttft') or 0) > 0:
        bad.append('%s probe has no positive median' % tag)

if bad:
    print('DEGENERATE: ' + '; '.join(bad))
    sys.exit(1)
"
}
