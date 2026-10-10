#!/usr/bin/env bash
# Falsification harness for scripts/bench/prefill.py.
#
# Why this needs proving at all: prefill.py publishes two README rows, and its
# central claim is that a SINGLE-length prompt_tokens/TTFT ratio is not a prefill
# rate. That claim came from measurement (a 15-token prompt measured a FASTER TTFT
# than a 347-token one, which prefill cannot do), and the harness once produced a
# prefill rate that was 2.3x wrong because of prefix-cache contamination between
# runs. A harness whose guards cannot be shown to fire is a guess, so every guard
# below is asserted in both directions.
#
# How this runs without a live engine: the argument-validation and metric-definition
# paths are checked by exit code, and the parts that must reach the network are
# checked against a dead port, where "rejected by the guard" and "failed at the
# socket" are distinguishable codes. Nothing here touches a serving endpoint.
#
# See docs/verification.md for the general discipline.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PF="$HERE/../bench/prefill.py"
BENCH="$(cd "$HERE/../bench" && pwd)"
DEAD_PORT="${DEAD_PORT:-1}"   # port 1 is not a vLLM; connection refused, fast

[[ -f "$PF" ]] || { echo "prefill-cases: missing $PF" >&2; exit 2; }

fails=0
ok()  { echo "  ok   $*"; }
bad() { echo "  FAIL $*"; fails=$((fails+1)); }

# $1=desc $2=expected-rc $3...=prefill.py args
expect_rc() {
  local desc="$1" want="$2"; shift 2
  local out rc
  out=$(python3 "$PF" --port "$DEAD_PORT" --model nonexistent "$@" 2>&1); rc=$?
  if [[ "$rc" == "$want" ]]; then
    ok "$desc (rc=$rc)"
  else
    bad "$desc: expected rc=$want, got $rc"
    printf '%s\n' "$out" | sed 's/^/         | /'
  fi
  LAST_OUT="$out"
}

echo "=== prefill.py falsification ==="

# 1. It must refuse to run without saying which mode it is in. Both modes need an
#    argument; silently defaulting to one would print a number for a measurement
#    nobody asked for.
expect_rc "no mode selected is rejected" 2

# 2. An unknown test name must be rejected rather than ignored.
expect_rc "unknown --test rejected" 2 --test definitely-not-a-test

# 3. The metric definitions must be shared with conc.py, not copied. A copy would
#    drift and the two harnesses would then disagree about what a test is -- the
#    exact class of bug that made a 96-vs-256 comparison look like a host fault.
shared=$(cd "$BENCH" && python3 -c '
import prefill, conc
same = prefill.TESTS is conc.TESTS and prefill.load_prompt is conc.load_prompt
print("SHARED" if same else "COPIED")' 2>&1)
if [[ "$shared" == "SHARED" ]]; then
  ok "TESTS and load_prompt are imported from conc.py, not duplicated"
else
  bad "prefill.py does not share the test table with conc.py ($shared)"
fi

# 4. The prompt for every test must resolve through that shared table.
#
#    Heredoc + explicit rc. This case's pass condition is EMPTY OUTPUT, and the earlier
#    `python3 -c '...\"prompt\"...'` form made Python die on the backslashes, print
#    nothing to a swallowed stderr, and thereby PASS. A check whose success condition is
#    "no output" must prove the program ran, or it rewards breakage.
missing=$(cd "$BENCH" && python3 - <<'PY' 2>&1
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
  ok "every test prompt exists on disk"
else
  bad "missing test prompt files: $missing"
fi

# 5. THE CENTRAL CLAIM. A single-length ratio must not be presented as a prefill
#    rate. The harness is required to carry the length sweep as the only thing that
#    yields prefill_tok_s, and to keep the floor-dominated low end out of the fit.
if cd "$BENCH" && python3 -c '
import prefill
assert hasattr(prefill, "run_length_sweep"), "no length sweep"
assert prefill.FIT_MIN_TTFT_MS > 0, "fit does not exclude the latency floor"
' 2>/dev/null; then
  ok "prefill rate requires the length sweep and excludes floor-dominated points"
else
  bad "prefill rate is derivable without the length sweep, or the fit includes the floor"
fi

# 6. The probe prompts must be unique ACROSS PROCESSES, not merely within one. This
#    is what a 2.3x wrong prefill rate cost: a per-process counter let a second run
#    rebuild byte-identical prompts and take cache hits on the longest ones.
uniq=$(cd "$BENCH" && python3 -c '
import subprocess, sys, os
code = ("import prefill;"
        "print(prefill.build_sweep_prompt(4096, 1))")
a = subprocess.run([sys.executable, "-c", code], cwd=os.getcwd(),
                   capture_output=True, text=True).stdout.strip()
b = subprocess.run([sys.executable, "-c", code], cwd=os.getcwd(),
                   capture_output=True, text=True).stdout.strip()
print("UNIQUE" if (a and b and a != b) else "REUSED")' 2>/dev/null)
if [[ "$uniq" == "UNIQUE" ]]; then
  ok "sweep prompts differ between processes (no cross-run cache hits)"
else
  bad "sweep prompts are reproducible across processes -- cache contamination possible"
fi

# 7. The salt must also differ between samples WITHIN a run, or one sample is a
#    prefix of another.
within=$(cd "$BENCH" && python3 -c '
import prefill
ps = {prefill.build_sweep_prompt(4096, i) for i in range(50)}
print("DISTINCT" if len(ps) == 50 else "COLLIDED")' 2>/dev/null)
if [[ "$within" == "DISTINCT" ]]; then
  ok "sweep prompts differ per sample within a run"
else
  bad "sweep prompt salts collide within a run"
fi

# 8. The slope fit must be able to say "I cannot fit this" rather than inventing a
#    number: fewer than two points, or a flat x-axis, must yield None.
#
#    The expected value is DERIVED, not typed. (0.2s, 100 tok) and (0.4s, 700 tok) is
#    600 tokens over 0.2 s = 3000 tok/s. The literal here used to say 2500.0, which was
#    an arithmetic slip in the harness, not a defect in the fit: hard-coding a number
#    you did not compute is how a check starts failing for reasons unrelated to the
#    code it is supposed to protect. The expectation is computed in Python below and
#    compared to the fit, so both sides come from the same arithmetic.
fit=$(cd "$BENCH" && python3 - <<'PY' 2>&1
import prefill
a = prefill.fit_slope([(0.2, 100.0)])
b = prefill.fit_slope([(0.2, 100.0), (0.2, 300.0)])
pts = [(0.2, 100.0), (0.4, 700.0)]
want = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])   # tokens per second
got = (prefill.fit_slope(pts) or {}).get("prefill_tok_s")
verdict = "OK" if (a is None and b is None and got == round(want, 1)) else "BAD"
print(verdict, a, b, got, want)
PY
)
fit_rc=$?
if [[ $fit_rc -ne 0 ]]; then
  bad "slope fit case could not run (rc=$fit_rc): $fit"
elif [[ "$fit" == OK* ]]; then
  ok "slope fit returns None for <2 points and a flat axis; matches derived slope"
else
  bad "slope fit degenerate-case behaviour wrong: got [$fit]"
fi

# 9. A test whose baseline drifted must be flagged rather than averaged.
stab=$(cd "$BENCH" && python3 -c '
import prefill
print(prefill.STABLE_MAX_SPREAD_MS)' 2>/dev/null)
if [[ -n "$stab" ]]; then
  ok "baseline stability threshold is defined ($stab ms)"
else
  bad "no baseline stability threshold -- drifting readings could be averaged"
fi

# 10. Warmup and measurement failures must be distinguishable from a real result,
#     and a dead endpoint must not print a table.
out=$(python3 "$PF" --port "$DEAD_PORT" --model x --test counting 2>/dev/null)
if [[ -z "${out// /}" ]]; then
  ok "dead endpoint prints no table on stdout (cannot be mistaken for a result)"
else
  bad "dead endpoint printed stdout: $out"
fi
if printf '%s' "${LAST_OUT:-}" | grep -q WARMUP_FAILED; then
  ok "warmup failure is named, not swallowed"
fi

# 11. Both modes must be discoverable, or the table is invisible to whoever runs it.
for flag in --length-sweep --test --rounds --iters; do
  if python3 "$PF" --help 2>&1 | grep -q -- "$flag"; then
    ok "$flag is discoverable in --help"
  else
    bad "$flag missing from --help"
  fi
done

echo
if [[ $fails -eq 0 ]]; then
  echo "falsify/prefill-cases: PASS -- prefill rate requires the slope, and its contamination guards fire"
  exit 0
fi
echo "falsify/prefill-cases: FAIL ($fails)"
exit 1
