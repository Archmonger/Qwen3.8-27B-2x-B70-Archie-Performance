#!/usr/bin/env bash
# Sustained-load soak, to look for the oneCCL IPC-handle wedge.
#
# Background: on TP2 the workers can hang inside oneCCL's ze_handle_manager
# IPC-handle exchange (a spin, not a crash) under load-shaped conditions, most
# often with a cold handle cache shortly after boot. It does NOT self-heal -- a
# wedged engine keeps returning /health 200 while generating nothing. The only
# mitigation found is CCL_SYCL_ALLREDUCE_TMP_BUF=1 (0/6 wedges vs 4/5 same-run
# controls, Fisher p=0.0152). Keep it set.
#
# What a PASS means here, and what it does not: N clean cycles lower the risk;
# they do not eliminate it, because the wedge is rare and load-shaped.
#
# Guards that keep a PASS honest (both were learned from harness bugs that
# reported a healthy engine as broken, and a killed run as healthy):
#   * rc=124 is the only code treated as a stall. rc=143 means THIS script killed
#     the load, so the cycle is recorded INCONCLUSIVE and the run aborts rather
#     than tallying a fake pass.
#   * a cycle only counts if it actually ran to its full duration.
#   * restart_count must not move.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE=fp8
CYCLES="${CYCLES:-6}"
CYCLE_S="${CYCLE_S:-600}"
# conc.py cell spec: concurrency:request-count pairs. conc=16 is above MAX_NUM_SEQS so
# requests queue, and conc=8 sits at the ceiling -- crossing it repeatedly is the shape
# that triggered the wedge. Overridable so the harness can be exercised quickly without
# editing the script (a guard nobody can run is a guard nobody has tested).
LOAD_CELLS="${LOAD_CELLS:-16:32,8:24}"
CTR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --container) CTR="${2:-}"; shift 2 ;;
    --cycles) CYCLES="${2:-}"; shift 2 ;;
    --cycle-seconds) CYCLE_S="${2:-}"; shift 2 ;;
    --load-cells) LOAD_CELLS="${2:-}"; shift 2 ;;
    *) echo "soak: unknown option $1" >&2; exit 2 ;;
  esac
done

[[ "$CYCLES" =~ ^[1-9][0-9]*$ ]] \
  || { echo "soak: --cycles must be a positive integer, got '$CYCLES'" >&2; exit 2; }
[[ "$CYCLE_S" =~ ^[1-9][0-9]*$ ]] \
  || { echo "soak: --cycle-seconds must be a positive integer, got '$CYCLE_S'" >&2; exit 2; }
# Reject a cell spec conc.py could not parse, rather than discovering it inside cycle 1
# after the deadline logic has already attributed the failure to the engine.
[[ "$LOAD_CELLS" =~ ^[1-9][0-9]*:[1-9][0-9]*(,[1-9][0-9]*:[1-9][0-9]*)*$ ]] \
  || { echo "soak: --load-cells must be conc:count[,conc:count...], got '$LOAD_CELLS'" >&2; exit 2; }

[[ -f "$HERE/../profiles/$PROFILE/$PROFILE.env" ]] \
  || { echo "soak: no such profile: $PROFILE" >&2; exit 2; }
_CLI_CTR="$CTR"
. "$HERE/lib/profile-env.sh"; load_env_file "$HERE/../profiles/$PROFILE/$PROFILE.env"
CTR="${_CLI_CTR:-$CONTAINER_NAME}"
# The profile exports SERVED_NAME; conc.py's --model is required and this script runs
# under `set -u`, so the served name has to be resolved here the way run-all.sh does it.
# It used to reference an unset $MNAME, which aborted cycle 1 with "MNAME: unbound
# variable" -- and since that is a clean exit rather than a stall, a run that never
# applied any load read as an engine problem instead of a bug in the driver.
MNAME="${SERVED_NAME:-secondary}"

restarts0=$(docker inspect "$CTR" --format '{{.RestartCount}}' 2>/dev/null || echo "?")
echo "[soak] $CTR port=$PORT cycles=$CYCLES x ${CYCLE_S}s (restart_count starts at $restarts0)"
timeout 10 curl -sf -m 8 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "[soak] ABORT: /health not answering"; exit 1; }

pass=0; inconclusive=0; stall=0
for c in $(seq 1 "$CYCLES"); do
  echo "[soak] --- cycle $c/$CYCLES (${CYCLE_S}s sustained at $LOAD_CELLS) ---"
  # The window is filled by REPEATING the load until the deadline, not by one burst.
  # A single burst used to take the whole 600s; with CUDA graphs on, decode roughly
  # doubled and the same burst drains in ~75s, so most cycles came back INCONCLUSIVE --
  # a soak that spends 87% of its time idle is not soaking, and the wedge this exists
  # to find is load-shaped. rc=0 now means "re-apply load", not "cycle over".
  # The batch crosses and lands on the ceiling repeatedly, which is the shape that
  # triggered the wedge. --reps would re-run the sweep serially; the load must be
  # concurrent. No --preserve-status: it reports 143 on a genuine timeout and destroys
  # the 124-vs-143 distinction this script depends on.
  #
  # /health is checked after EVERY load attempt, and completed rounds are COUNTED. The
  # counting is what makes the deadline pass honest: the wedge is a hang that keeps
  # answering /health while generating nothing, so without a progress requirement
  # "reached the full 600s" would be equally satisfiable by an engine that never
  # finished a single request -- a pass condition that a hang satisfies is not a pass
  # condition. A cycle therefore needs BOTH the elapsed window AND at least one load
  # that completed under its own power.
  t_start=$(date +%s)
  t_end=$(( t_start + CYCLE_S ))
  rc=0; stalled=0; rounds=0
  while :; do
    now=$(date +%s)
    remaining=$(( t_end - now ))
    if [[ $remaining -le 0 ]]; then rc=124; break; fi
    timeout -k 30 "$remaining" \
      python3 "$HERE/bench/conc.py" --port "$PORT" --model "$MNAME" \
        --cells "$LOAD_CELLS" --max-tokens 256
    rc=$?
    if ! timeout 10 curl -sf -m 8 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "[soak] cycle $c: STALL -- /health stopped answering (rc=$rc)"
      stalled=1; break
    fi
    # 124 = the deadline fired, which is the intended end of a cycle.
    # 143 = operator kill. Anything else non-zero = load failed. All three break.
    [[ $rc -ne 0 ]] && break
    rounds=$(( rounds + 1 ))
    # rc=0: the load drained early. Keep hammering until the window closes.
  done
  elapsed=$(( $(date +%s) - t_start ))
  restarts=$(docker inspect "$CTR" --format '{{.RestartCount}}' 2>/dev/null || echo "?")
  if [[ $stalled -eq 1 ]]; then
    stall=$((stall+1)); continue
  fi
  case "$rc" in
    124)
      # The deadline fired. That is the intended way out of a cycle, but it is NOT on
      # its own evidence of a healthy engine: an engine that accepted requests and then
      # hung inside oneCCL also reaches the deadline, keeps answering /health, and
      # completes nothing. So the pass requires completed load as well as elapsed time.
      if [[ $elapsed -ge $(( CYCLE_S * 9 / 10 )) && $rounds -ge 1 ]]; then
        echo "[soak] cycle $c: ran to duration (${elapsed}s, $rounds load round(s) completed), engine healthy"
        pass=$((pass+1))
      elif [[ $rounds -eq 0 ]]; then
        # Not a stall the health check could see, and not a pass: no request ever
        # finished, which is what a wedge looks like from the outside.
        echo "[soak] cycle $c: FAILED -- ${elapsed}s elapsed but ZERO load rounds completed (probable wedge; check docker logs $CTR)"
        stall=$((stall+1))
      else
        echo "[soak] cycle $c: INCONCLUSIVE -- rc=124 but only ${elapsed}s of ${CYCLE_S}s ($rounds round(s))"
        inconclusive=$((inconclusive+1))
      fi ;;
    0)
      # Unreachable under the repeat loop unless the window was smaller than one load
      # (a short --cycle-seconds). Kept so a fast cycle is reported, not counted as a
      # sustained soak.
      echo "[soak] cycle $c: INCONCLUSIVE -- window closed before sustaining load (${elapsed}s of ${CYCLE_S}s, $rounds round(s))"
      inconclusive=$((inconclusive+1)) ;;
    143) echo "[soak] cycle $c: INCONCLUSIVE -- load was killed (rc=143), not a stall and not a pass"
           inconclusive=$((inconclusive+1)) ;;
    *)     echo "[soak] cycle $c: FAILED rc=$rc after $rounds round(s) (check docker logs $CTR)"
           inconclusive=$((inconclusive+1)) ;;
  esac
  if [[ "$restarts" != "$restarts0" ]]; then
    echo "[soak] FAIL: restart_count moved $restarts0 -> $restarts"
    stall=$((stall+1))
  fi
done

echo
echo "[soak] cycles passed=$pass inconclusive=$inconclusive stall/failed=$stall of $CYCLES"
if [[ $stall -gt 0 ]]; then
  echo "[soak] FAIL: the wedge or a restart was observed. Do not ship this config."
  exit 1
fi
if [[ $inconclusive -gt 0 ]]; then
  echo "[soak] INCONCLUSIVE: $inconclusive cycle(s) did not run to completion (including any operator kill)."
  echo "[soak] A PASS cannot come from a killed run -- re-run the affected cycles."
  exit 2
fi
echo "[soak] PASS: $pass/$CYCLES sustained cycles, 0 wedges, 0 restarts. Six clean cycles lower"
echo "[soak] the risk; they do not eliminate it. Keep CCL_SYCL_ALLREDUCE_TMP_BUF=1."
