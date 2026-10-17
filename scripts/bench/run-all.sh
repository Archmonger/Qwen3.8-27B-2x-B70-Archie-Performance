#!/usr/bin/env bash
# Reproduce the throughput table in the README against a running endpoint.
#
# Read this before quoting a number:
#  * Run it on an IDLE engine, one cell at a time. Two measurements sharing the
#    box contaminate each other, and both look fine.
#  * The grid's top cell sits at the MAX_NUM_SEQS cap, and it is the cell that
#    decides these comparisons. MAX_NUM_SEQS=10 means a busy engine is always at
#    the 10-request cap, and the graph ceiling is MAX_NUM_SEQS x (1+k). Measuring
#    below the cap is what hid a 38% batched loss across two deploys: the step
#    under the ceiling fits in a captured graph while the capped step does not, so
#    it silently falls back to eager. See docs/cudagraph-stride-defect.md.
#  * The box drifts ~5% per 10 boots on the natural probe. Any claimed delta
#    under ~3 tok/s on the natural metric needs repeats, and a real A/B must boot
#    both arms back-to-back with the order alternated.
#  * --max-tokens is part of the metric, not a knob: aggregate_tok_s moves ~15%
#    between 96 and 256 at conc=10 while single stream barely moves, because the
#    uncounted prefill of a fixed prompt is amortised over max_tokens. Each named
#    test in conc.py pins its own budget next to its prompt, so nothing here passes
#    a budget by hand; conc.py refuses a value other than the pinned one unless you
#    pass --confirm-noncanonical.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE=fp8
TESTS=""
GRID_CELLS=""
REPS=1
DO_SUITE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)     PROFILE="${2:-}"; shift 2 ;;
    --tests)       TESTS="${2:-}"; shift 2 ;;
    --cells)       GRID_CELLS="${2:-}"; shift 2 ;;
    --reps)        REPS="${2:-}"; shift 2 ;;
    --no-suite)    DO_SUITE=0; shift ;;
    *) echo "run-all: unknown option $1" >&2; exit 2 ;;
  esac
done

[[ "$REPS" =~ ^[0-9]+$ && "$REPS" -ge 1 ]] \
  || { echo "run-all: --reps must be a positive integer, got '$REPS'" >&2; exit 2; }

[[ -f "$HERE/../../profiles/$PROFILE/$PROFILE.env" ]] \
  || { echo "run-all: no such profile: $PROFILE" >&2; exit 2; }
. "$HERE/../lib/profile-env.sh"; load_env_file "$HERE/../../profiles/$PROFILE/$PROFILE.env"
# Same host KV-tier overlay setup.sh applied, so the measured table is recorded
# under the tier the container actually booted with rather than the repo default.
# See scripts/lib/profile-env.sh.
load_host_overlay "$HERE/../../profiles/$PROFILE/$PROFILE.env" \
  || { echo "run-all: host env overlay is invalid (see above)" >&2; exit 2; }
PORT="${PORT:-8103}"; MNAME="${SERVED_NAME:-secondary}"

# The grid the README table is built from: one test x one concurrency per cell, and
# the published cell is the median over REPS. The top cell is derived from
# MAX_NUM_SEQS so the sweep measures a busy engine rather than a mostly-idle one on
# any profile; the second sits under the cap and the third is single stream. Each
# cell is conc:total_requests, with enough requests to run several waves.
if [[ -z "$TESTS" ]]; then TESTS="counting planning prose coding"; fi
if [[ -z "$GRID_CELLS" ]]; then
  cap=$(( ${MAX_NUM_SEQS:-10} ))
  mid=$(( cap / 2 ))
  [[ "$mid" -ge 1 ]] || mid=1
  GRID_CELLS="${cap}:$((cap*3)),${mid}:$((mid*4)),1:6"
  echo "[bench] cells derived from MAX_NUM_SEQS=$cap: $GRID_CELLS"
fi

timeout 10 curl -sf -m 8 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "[bench] ABORT: /health not answering on $PORT"; exit 1; }

echo "[bench] port $PORT model $MNAME"

# Test order alternates with the rep so a drift across the run is shared by every
# cell instead of landing on whichever ran last -- the same reason a paired A/B
# alternates arm order rather than running each arm once.
read -r -a TEST_ARR <<< "$TESTS"
NTEST=${#TEST_ARR[@]}
[[ "$NTEST" -ge 1 ]] || { echo "run-all: --tests parsed to nothing" >&2; exit 2; }
for r in $(seq 1 "$REPS"); do
  [[ "$REPS" -gt 1 ]] && echo "[bench] --- rep $r/$REPS ---"
  for i in $(seq 0 $((NTEST - 1))); do
    if [[ $((r % 2)) -eq 1 ]]; then
      t="${TEST_ARR[$i]}"
    else
      t="${TEST_ARR[$((NTEST - 1 - i))]}"
    fi
    echo "[bench] grid test=$t rep=$r cells=$GRID_CELLS"
    python3 "$HERE/conc.py" --port "$PORT" --model "$MNAME" \
            --test "$t" --cells "$GRID_CELLS" \
      || echo "[bench] grid sweep failed for test=$t"
  done
done

echo "[bench] prompt processing + TTFT (the two non-token-generation rows)"
# The sweep is a SEPARATE invocation from the per-test TTFT run, and both are needed:
# the README's Prompt Processing row is the slope from --length-sweep, while its TTFT
# row is the per-test median. Running only one of them leaves a published row that the
# documented reproduce command cannot produce -- which is how a table quietly becomes
# un-reproducible one row at a time.
python3 "$HERE/prefill.py" --port "$PORT" --model "$MNAME" --length-sweep \
        --rounds "${PREFILL_ROUNDS:-5}" \
  || echo "[bench] prompt-processing sweep failed"
python3 "$HERE/prefill.py" --port "$PORT" --model "$MNAME" \
        --test counting --test planning --test prose --test coding \
        --iters "${PREFILL_ITERS:-8}" --warmup 2 \
  || echo "[bench] prefill probe failed"

echo "[bench] supporting checks (recorded in docs/recorded-metrics.md, not the README)"
python3 "$HERE/probe.py" --url "http://127.0.0.1:$PORT/v1/completions" \
        --model "$MNAME" --prompt "1 2" --max-tokens 16 --iters 10 --warmup 3 --label C1 \
  || echo "[bench] C1 probe failed"
python3 "$HERE/probe.py" --url "http://127.0.0.1:$PORT/v1/completions" \
        --model "$MNAME" --prompt "$(cat "$HERE/../../data/prompt-planning.txt")" \
        --max-tokens 96 --iters 6 --warmup 2 --label NAT \
  || echo "[bench] natural probe failed"

if [[ "$DO_SUITE" -eq 1 ]]; then
  echo "[bench] realistic suite (tok_s_1_100_after_ttft, the canonical metric)"
  PORT="$PORT" SERVED_NAME="$MNAME" python3 "$HERE/suite.py" || echo "[bench] suite failed"
fi

echo "[bench] done. Quote the metric name with any number you report -- engine-log"
echo "[bench] 'Avg generation throughput' is an aggregate and is not comparable."
