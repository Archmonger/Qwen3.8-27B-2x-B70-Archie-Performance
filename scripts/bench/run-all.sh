#!/usr/bin/env bash
# Reproduce the throughput table in the README against a running endpoint.
#
# Read this before quoting a number:
#  * Run it on an IDLE engine, one cell at a time. Two measurements sharing the
#    box contaminate each other, and both look fine.
#  * Measure at the MAX_NUM_SEQS cap, not below it. MAX_NUM_SEQS=10 means a busy
#    engine is always at the 10-request cap, and the graph ceiling is
#    MAX_NUM_SEQS x (1+k). Measuring at conc=8 is what hid a 38% batched loss
#    across two deploys: the conc=8 step (64 tokens) fits under the ceiling while
#    the conc=10 step (80 tokens) does not, so it silently falls back to eager.
#  * The box drifts ~5% per 10 boots on the natural probe. Any claimed delta
#    under ~3 tok/s on the natural metric needs repeats, and a real A/B must boot
#    both arms back-to-back with the order alternated.
#  * A 14% peak-to-trough spread on a single live engine is normal.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE=fp8
CELLS=""
REPS=1
DO_SUITE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --cells)   CELLS="${2:-}"; shift 2 ;;
    --reps)    REPS="${2:-}"; shift 2 ;;
    --no-suite) DO_SUITE=0; shift ;;
    *) echo "run-all: unknown option $1" >&2; exit 2 ;;
  esac
done

[[ -f "$HERE/../../profiles/$PROFILE/$PROFILE.env" ]] \
  || { echo "run-all: no such profile: $PROFILE" >&2; exit 2; }
. "$HERE/../lib/profile-env.sh"; load_env_file "$HERE/../../profiles/$PROFILE/$PROFILE.env"
PORT="${PORT:-8103}"; MNAME="${SERVED_NAME:-secondary}"

# The default cells straddle the ceiling so the cliff, if any, is visible. Each
# cell is conc:total_requests.
if [[ -z "$CELLS" ]]; then
  stride=$(( 1 + ${NUM_SPEC_TOKENS:-7} ))
  cap=$(( ${MAX_NUM_SEQS:-10} ))
  CELLS="1:6,8:$((8*3)),${cap}:$((cap*3)),$((cap+2)):$((cap+2))"
  echo "[bench] cells chosen from MAX_NUM_SEQS=$cap and stride=$stride: $CELLS"
  echo "[bench]   the ${cap}-request batch is $((cap*stride)) tokens; ceiling is ${CEILING:-?}"
fi

timeout 10 curl -sf -m 8 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "[bench] ABORT: /health not answering on $PORT"; exit 1; }

echo "[bench] port $PORT model $MNAME"
for r in $(seq 1 "$REPS"); do
  [[ $REPS -gt 1 ]] && echo "[bench] --- rep $r/$REPS ---"
  echo "[bench] concurrency sweep (aggregate tok/s)"
  python3 "$HERE/conc.py" --port "$PORT" --model "$MNAME" \
          --cells "$CELLS" --max-tokens 256 || echo "[bench] conc sweep failed"
done

echo "[bench] isolated probes (C1 counting + natural, the rows the README quotes)"
python3 "$HERE/probe.py" --url "http://127.0.0.1:$PORT/v1/completions" \
        --model "$MNAME" --prompt "1 2" --max-tokens 16 --iters 10 --warmup 3 --label C1 \
  || echo "[bench] C1 probe failed"
python3 "$HERE/probe.py" --url "http://127.0.0.1:$PORT/v1/completions" \
        --model "$MNAME" --prompt "$(cat "$HERE/../../data/prompt512.txt")" \
        --max-tokens 96 --iters 6 --warmup 2 --label NAT \
  || echo "[bench] natural probe failed"

if [[ $DO_SUITE -eq 1 ]]; then
  echo "[bench] realistic suite (tok_s_1_100_after_ttft, the canonical metric)"
  PORT="$PORT" SERVED_NAME="$MNAME" python3 "$HERE/suite.py" || echo "[bench] suite failed"
fi

echo "[bench] done. Quote the metric name with any number you report -- engine-log"
echo "[bench] 'Avg generation throughput' is an aggregate and is not comparable."
