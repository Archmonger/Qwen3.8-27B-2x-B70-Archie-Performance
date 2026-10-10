#!/usr/bin/env bash
# Attest that the running container IS the measured configuration, then prove it
# works. Order is deliberate: /health only says a process is listening, and this
# image has no HEALTHCHECK, so a config that garbles text still returns 200.
#
# Attestation reads the engine's own printed config from the container log via
# scripts/lib/attest.sh -- not the env file. See that file for what each rule
# protects against, and scripts/falsify/verify-cases.sh for the proof that each
# rule can fail. A verdict path that cannot fail is not a test:
# docs/verification.md.
#
# Usage:  scripts/verify.sh [--profile fp8] [--container NAME] [--port N]
#                           [--model NAME] [--lenient] [--skip-bench]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE=fp8
CTR="" PORT="" MNAME="" STRICT=1 SKIP_BENCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)    PROFILE="${2:-}"; shift 2 ;;
    --container)  CTR="${2:-}"; shift 2 ;;
    --port)       PORT="${2:-}"; shift 2 ;;
    --model)      MNAME="${2:-}"; shift 2 ;;
    --lenient)    STRICT=0; shift ;;
    --skip-bench) SKIP_BENCH=1; shift ;;
    *) echo "verify: unknown option $1" >&2; exit 2 ;;
  esac
done

PROF_ENV="$HERE/../profiles/$PROFILE/$PROFILE.env"
[[ -f "$PROF_ENV" ]] || { echo "verify: no such profile: $PROFILE" >&2; exit 2; }
# Capture CLI values first: load_env_file exports the profile's own PORT/SERVED_NAME
# and would otherwise overwrite them.
_CLI_CTR="$CTR"; _CLI_PORT="$PORT"; _CLI_MNAME="$MNAME"
. "$HERE/lib/profile-env.sh"; load_env_file "$PROF_ENV"
. "$HERE/lib/attest.sh"
CTR="${_CLI_CTR:-${CONTAINER_NAME:-}}"
PORT="${_CLI_PORT:-${PORT:-8103}}"
MNAME="${_CLI_MNAME:-${SERVED_NAME:-secondary}}"

fails=0
bad() { echo "  FAIL $*"; fails=$((fails+1)); }
good(){ echo "  ok   $*"; }

echo "=== verify: ${CTR:-<none>} (port $PORT, model $MNAME) ==="

# --- 1. liveness -------------------------------------------------------------
echo "[1/5] /health"
if ! timeout 10 curl -sf -m 8 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "  FAIL /health not answering on $PORT (docker logs ${CTR:-?} | tail -40)"
  exit 1
fi
good "healthy"
if [[ -n "$CTR" ]]; then
  docker ps --format '{{.Names}}' | grep -qx "$CTR" || bad "container $CTR is not running"
fi

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
LOG="$WORK/engine.log"; : > "$LOG"
BOOTED_AT=""
if [[ -n "$CTR" ]]; then
  # Timestamps, because boot_slice needs them to tell one boot from the next. Plain
  # `docker logs` has no per-line time and cannot be sliced.
  docker logs --timestamps "$CTR" > "$WORK/raw.log" 2>&1 \
    || { echo "  FAIL cannot read logs of $CTR"; exit 1; }
  BOOTED_AT=$(docker inspect -f '{{.State.StartedAt}}' "$CTR" 2>/dev/null)
  RESTARTS=$(docker inspect -f '{{.RestartCount}}' "$CTR" 2>/dev/null)
  if ! boot_slice "$WORK/raw.log" "$BOOTED_AT" "$LOG"; then
    # Empty slice means the timestamps do not line up with StartedAt, not that the
    # boot is clean. Fall back to the whole log so the check can only get stricter.
    echo "  WARN boot slice empty (StartedAt=${BOOTED_AT:-missing}); scanning the full log"
    cat "$WORK/raw.log" > "$LOG"
  fi
  # A restart is not something attestation should absorb silently: it means an earlier
  # boot died, and which boot produced the numbers you are about to quote matters.
  if [[ -n "$RESTARTS" && "$RESTARTS" != "0" ]]; then
    echo "  NOTE container has restarted $RESTARTS time(s); attesting only this boot" \
         "(started ${BOOTED_AT:-unknown})"
  fi
fi

# --- 2. attestation ---------------------------------------------------------
echo "[2/5] config attestation (engine's own log, not the env file)"
want_ceil=$(( ${MAX_NUM_SEQS:-10} * (1 + ${NUM_SPEC_TOKENS:-7}) ))
want_k="${NUM_SPEC_TOKENS:-}"; want_kv="${KV_CACHE_DTYPE:-}"
attest "$LOG" "$want_ceil" "$want_k" "$want_kv" > "$WORK/verdicts" 2>/dev/null
attest_labels "$want_ceil" "$want_k" "$want_kv" > "$WORK/labels"
# Join on the rule name. A rule with a verdict but no label (or vice versa) is a
# harness edit that only half landed, so it is reported rather than dropped.
while IFS='|' read -r _ name verdict detail; do
  label="$(awk -F'|' -v n="$name" '$2==n{print $3}' "$WORK/labels")"
  if [[ -z "$label" ]]; then bad "rule '$name' has no label (attest_labels out of sync)"; continue; fi
  if [[ "$verdict" == ok ]]; then good "$label"; else bad "$label -- $detail"; fi
done < "$WORK/verdicts"
if [[ ! -s "$WORK/verdicts" ]]; then
  bad "attestation produced no verdicts at all (reading an empty log is not a pass)"
fi

# --- 3. correctness ---------------------------------------------------------
echo "[3/5] correctness"
if PORT="$PORT" SERVED_NAME="$MNAME" bash "$HERE/bench/correctness.sh"; then good "correctness"; else bad "correctness"; fi

# --- 4. spec-decode attestation --------------------------------------------
echo "[4/5] acceptance"
if PORT="$PORT" SERVED_NAME="$MNAME" bash "$HERE/bench/acceptance.sh"; then good "acceptance"; else bad "acceptance"; fi

# --- 5. serving surface -----------------------------------------------------
echo "[5/5] serving surface"
if PORT="$PORT" SERVED_NAME="$MNAME" timeout 900 python3 "$HERE/bench/surface.py" "$PORT" "$MNAME"; then
  good "serving surface"
else
  bad "serving surface"
fi

if [[ $SKIP_BENCH -eq 0 ]]; then
  echo
  echo "not run here: throughput. Benchmarks measure something different if anything"
  echo "else is running. Use scripts/bench/run-all.sh on an idle engine."
fi

echo
if [[ $fails -eq 0 ]]; then
  echo "VERIFY: PASS -- this container is provably the measured configuration and it works"
  exit 0
fi
echo "VERIFY: FAIL ($fails check(s))"
[[ $STRICT -eq 1 ]] && exit 1 || exit 0
