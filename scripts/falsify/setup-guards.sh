#!/usr/bin/env bash
# Falsification harness for setup.sh's preflight guards.
#
# Each case mutates profiles/fp8/fp8.env, runs `setup.sh --dry-run`, and requires
# the guard to FIRE. A guard that never fires is worse than no guard: the preflight
# reads as "the box was checked".
#
# The profile is backed up to a temp dir and restored on every exit path, including
# failure paths. Run this before trusting setup.sh, and after editing any guard.
#
# Usage:  scripts/falsify/setup-guards.sh
# Exit 0 only if every guard fires and the unmutated profile is clean.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ENVF="$REPO/profiles/fp8/fp8.env"
[[ -f "$ENVF" ]] || { echo "falsify/setup-guards: no profile at $ENVF" >&2; exit 2; }

BK="$(mktemp -d)/fp8.env"
cp "$ENVF" "$BK"
restore() { cp "$BK" "$ENVF"; }
trap restore EXIT

fails=0
run_case(){ # $1=name $2=expect(fire|pass|warn) $3=pattern $4=mutate-function
  local name="$1" mode="$2" pat="$3" fn="$4" out rc first_fatal
  restore; "$fn"
  out="$("$REPO/setup.sh" --dry-run 2>&1)"; rc=$?
  first_fatal="$(grep -m1 'FATAL' <<<"$out")"
  case "$mode" in
    fire)
      if grep -qF -- "$pat" <<<"$out"; then
        echo "ok   $name: guard fired"
      else
        echo "FAIL $name: guard did NOT fire (rc=$rc)"
        # Report what it died on instead. A model dir missing on this box, for
        # example, means preflight never reached the guard under test -- a fixture
        # problem, and never a passing case.
        if [[ -n "$first_fatal" ]]; then
          echo "       instead tripped: ${first_fatal#\[setup\] }"
        else
          echo "       and tripped nothing at all"
        fi
        fails=$((fails+1))
      fi ;;
    pass)
      # The baseline must be clean, or every "guard fired" line below means nothing.
      # Assert the exit code rather than grepping for a success string: a message can
      # be reworded, and a harness that only tracks prose silently stops testing.
      if [[ -n "$first_fatal" ]]; then
        echo "FAIL $name: the UNMUTATED profile trips: ${first_fatal#\[setup\] }"
        echo "       fix the profile or the preflight before trusting the cases below"
        fails=$((fails+1))
      elif [[ $rc -ne 0 ]]; then
        echo "FAIL $name: dry run exited $rc on an unmutated profile"
        tail -3 <<<"$out" | sed 's/^/       /'
        fails=$((fails+1))
      else
        echo "ok   $name: baseline clean"
      fi ;;
    warn)
      if grep -qF -- "$pat" <<<"$out" && [[ -z "$first_fatal" ]]; then
        echo "ok   $name: warned without aborting"
      elif [[ -n "$first_fatal" ]]; then
        echo "FAIL $name: expected a warning, got FATAL: ${first_fatal#\[setup\] }"
        fails=$((fails+1))
      else
        echo "FAIL $name: expected a warning, got nothing"
        fails=$((fails+1))
      fi ;;
  esac
}

m_none()        { :; }
m_ceiling()     { sed -i 's/^CEILING=80/CEILING=72/' "$ENVF"; }
m_ceiling_args(){ sed -i 's/"max_cudagraph_capture_size":80/"max_cudagraph_capture_size":72/' "$ENVF"; }
m_seq8()        { sed -i 's/^MAX_NUM_SEQS=10/MAX_NUM_SEQS=8/' "$ENVF"; }
m_k6()          { sed -i 's/^NUM_SPEC_TOKENS=7/NUM_SPEC_TOKENS=6/' "$ENVF"; }
m_sched()       { printf '\nSPEC_TOKENS_SCHEDULE=[[1,1,7],[2,64,4]]\n' >> "$ENVF"; }
m_i915()        { sed -i 's/^B70_BDFS=.*/B70_BDFS=0000:43:00.0,0000:83:00.0/' "$ENVF"; }
m_onebdf()      { sed -i 's/^B70_BDFS=.*/B70_BDFS=0000:03:00.0/' "$ENVF"; }
m_nograph()     { sed -i 's/"cudagraph_mode":"FULL_DECODE_ONLY"/"cudagraph_mode":"NONE"/' "$ENVF"; }
m_dflash()      { sed -i 's/^SPEC_METHOD=dspark/SPEC_METHOD=dflash/' "$ENVF"; }

echo "=== setup.sh guard falsification ==="
run_case "unmutated profile       " pass ""           m_none
run_case "CEILING=72              " fire "graph-ceiling rule violated" m_ceiling
run_case "ceiling in args =72     " fire "EXTRA_VLLM_ARGS max_cudagraph_capture_size" m_ceiling_args
run_case "MAX_NUM_SEQS=8 ceil 80  " fire "graph-ceiling rule violated" m_seq8
run_case "k=6, ceiling left at 80 " fire "graph-ceiling rule violated" m_k6
run_case "SPEC_TOKENS_SCHEDULE set" fire "SPEC_TOKENS_SCHEDULE is set" m_sched
run_case "B70 list includes A310  " fire "expected xe" m_i915
run_case "only one BDF            " fire "exactly 2 PCI BDFs" m_onebdf
run_case "cudagraph_mode NONE     " fire "must set cudagraph_mode FULL_DECODE_ONLY" m_nograph
run_case "SPEC_METHOD=dflash      " warn "does not boot with this drafter" m_dflash
restore

echo
if [[ $fails -eq 0 ]]; then
  echo "falsify/setup-guards: PASS -- every guard can fire and the baseline is clean"
  exit 0
fi
echo "falsify/setup-guards: FAIL ($fails problem(s)) -- do not trust setup.sh until this is green"
exit 1
