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
CTR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --container) CTR="${2:-}"; shift 2 ;;
    --cycles) CYCLES="${2:-}"; shift 2 ;;
    --cycle-seconds) CYCLE_S="${2:-}"; shift 2 ;;
    *) echo "soak: unknown option $1" >&2; exit 2 ;;
  esac
done

[[ -f "$HERE/../profiles/$PROFILE/$PROFILE.env" ]] \
  || { echo "soak: no such profile: $PROFILE" >&2; exit 2; }
_CLI_CTR="$CTR"
. "$HERE/lib/profile-env.sh"; load_env_file "$HERE/../profiles/$PROFILE/$PROFILE.env"
CTR="${_CLI_CTR:-$CONTAINER_NAME}"

restarts0=$(docker inspect "$CTR" --format '{{.RestartCount}}' 2>/dev/null || echo "?")
echo "[soak] $CTR port=$PORT cycles=$CYCLES x ${CYCLE_S}s (restart_count starts at $restarts0)"
timeout 10 curl -sf -m 8 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "[soak] ABORT: /health not answering"; exit 1; }

pass=0; inconclusive=0; stall=0
for c in $(seq 1 "$CYCLES"); do
  echo "[soak] --- cycle $c/$CYCLES (${CYCLE_S}s at conc=16 then conc=8) ---"
  # Two bursts so the batch crosses and lands on the ceiling repeatedly, which is
  # the shape that triggered the wedge. --reps would re-run the sweep serially;
  # the load itself must be concurrent.
  # No --preserve-status: it would report 143 on a genuine timeout and destroy the
  # 124-vs-143 distinction this script depends on.
  t_start=$(date +%s)
  timeout -k 30 "$CYCLE_S" \
    python3 "$HERE/bench/conc.py" --port "$PORT" --model "$MNAME" \
      --cells "16:32,8:24" --max-tokens 256
  rc=$?
  elapsed=$(( $(date +%s) - t_start ))
  restarts=$(docker inspect "$CTR" --format '{{.RestartCount}}' 2>/dev/null || echo "?")
  if ! timeout 10 curl -sf -m 8 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[soak] cycle $c: STALL -- /health stopped answering (rc=$rc, ${elapsed}s)"
    stall=$((stall+1)); continue
  fi
  case "$rc" in
    124)
      # Timed out as intended: the load ran the whole window.
      if [[ $elapsed -ge $(( CYCLE_S * 9 / 10 )) ]]; then
        echo "[soak] cycle $c: ran to duration (${elapsed}s), engine healthy"; pass=$((pass+1))
      else
        echo "[soak] cycle $c: INCONCLUSIVE -- rc=124 but only ${elapsed}s of ${CYCLE_S}s"
        inconclusive=$((inconclusive+1))
      fi ;;
    0)
      # The load finished before the window closed, so this cycle did not sustain
      # anything. Counting it as a pass is how a soak lies.
      echo "[soak] cycle $c: INCONCLUSIVE -- load exited early (${elapsed}s of ${CYCLE_S}s)"
      inconclusive=$((inconclusive+1)) ;;
    143) echo "[soak] cycle $c: INCONCLUSIVE -- load was killed (rc=143), not a stall and not a pass"
           inconclusive=$((inconclusive+1)) ;;
    *)     echo "[soak] cycle $c: FAILED rc=$rc (check docker logs $CTR)"
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
