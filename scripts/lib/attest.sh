#!/usr/bin/env bash
# Config attestation, shared by scripts/verify.sh and scripts/falsify/verify-cases.sh.
#
# Extracted so the falsification harness exercises the SAME extraction and
# comparison rules the live check uses. A harness that re-implements the logic
# drifts from it, and a drifting harness keeps passing while the real check is
# broken -- the failure class this project has actually been bitten by.
#
# Usage:  source scripts/lib/attest.sh
#         attest <engine-log-file> <want_ceiling> <want_k> [kv_dtype]
#
# Prints one "rule|<name>|<ok|bad>|<detail>" line per check and returns the number
# of failures (capped at 125 so a return value stays usable as an exit code).
# attest_labels <want_ceiling> <want_k> [kv_dtype] prints "rule|<name>|<label>".
#
# It reads the ENGINE'S OWN printed config, never the env file: an env var vLLM
# ignores, or one an image rebuild silently dropped, must not be attestable.

attest_labels() { # $1=want_ceiling $2=want_k $3=kv
  local want_k="${2:-}" kv="${3:-}"
  echo "rule|ceil|max_cudagraph_capture_size=$1 (MAX_NUM_SEQS x (1+k))"
  echo "rule|ladder|capture ladder contains the $1-token saturated batch"
  echo "rule|cg|cudagraph_mode=FULL_DECODE_ONLY"
  [[ -n "$want_k" ]] && echo "rule|k|num_speculative_tokens=$want_k (static)"
  echo "rule|meth|speculative method=dspark"
  echo "rule|sched|no dynamic speculative schedule"
  echo "rule|capture|graph capture ran"
  echo "rule|fatal|no fatal signature in the engine log"
  echo "rule|v2|V2 model runner (required by dspark)"
  [[ -n "$kv" ]] && echo "rule|kv|KV cache dtype $kv"
}

attest() { # $1=log $2=want_ceiling $3=want_k (empty = skip) $4=kv dtype (empty = skip)
  local LOG="$1" want_ceil="$2" want_k="${3:-}" kv="${4:-}"
  local ceil cg nsp meth sched sizes maxsize cap nbad=0

  # The ceiling number alone is not proof: vLLM builds the capture ladder FROM it,
  # and if the ladder does not contain the saturated batch size then that batch is
  # still eager. This is the check that caught the 72-vs-80 defect.
  ceil=$(grep -oE 'max_cudagraph_capture_size.: [0-9]+'          "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
  cg=$(grep -oE 'cudagraph_mode.: <CUDAGraphMode[^>]*>'          "$LOG" 2>/dev/null | tail -1)
  nsp=$(grep -oE 'num_speculative_tokens.: [0-9]+'               "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
  meth=$(grep -oE "method.: .[a-z]+"                             "$LOG" 2>/dev/null | tail -1 | sed "s/.*'//")
  sched=$(grep -oE 'num_speculative_tokens_per_batch_size.: \[\[.{0,40}' "$LOG" 2>/dev/null | tail -1)
  sizes=$(grep -oE 'cudagraph_capture_sizes.: \[[0-9, ]+\]'      "$LOG" 2>/dev/null | tail -1)
  cap=$(grep -oE "Graph capturing finished in [0-9.]+ secs, took [0-9.]+ GiB" "$LOG" 2>/dev/null | sort -u)
  maxsize=$(printf '%s' "$sizes" | tr -d 'a-z_[] ' | tr ',' '\n' | grep -E '^[0-9]+$' | sort -n | tail -1)

  # Machine-readable output: "rule|<name>|<ok|bad>|<detail>". verify.sh owns the
  # human labels, scripts/falsify/verify-cases.sh owns the verdict discrimination,
  # and neither has to re-implement the extraction rules to do it.
  ok()  { echo "rule|$1|ok|"; }
  bad() { echo "rule|$1|bad|${2:-}"; nbad=$((nbad+1)); }

  # 1. graph ceiling equals MAX_NUM_SEQS x (1+k), as computed by the caller
  [[ "$ceil" == "$want_ceil" ]] && ok ceil || bad ceil "max_cudagraph_capture_size=${ceil:-missing}, expected $want_ceil"
  # 2. the capture ladder actually contains the saturated batch size
  [[ "$maxsize" == "$want_ceil" ]] && ok ladder || bad ladder "capture ladder tops out at ${maxsize:-none}, must contain $want_ceil ($sizes)"
  # 3. graphs on, and the full-decode kind
  [[ "$cg" == *"FULL_DECODE_ONLY"* ]] && ok cg || bad cg "cudagraph_mode=${cg:-missing}"
  # 4. static k equals the measured value (skipped when no k is expected)
  if [[ -n "$want_k" ]]; then
    [[ "$nsp" == "$want_k" ]] && ok k || bad k "num_speculative_tokens=${nsp:-missing}, expected $want_k"
  fi
  # 5. dspark, not dflash (dflash does not boot with the published drafter on 0.26)
  [[ "$meth" == "dspark" ]] && ok meth || bad meth "speculative method=${meth:-missing}"
  # 6. no dynamic schedule: with graphs on it garbles output at healthy counters
  [[ -z "$sched" ]] && ok sched || bad sched "dynamic schedule active: $sched"
  # 7. capture actually ran -- absent means every shape is eager
  [[ -n "$cap" ]] && ok capture || bad capture "no 'Graph capturing finished' line -- nothing was graphed"
  # 8. no fatal signature masquerading as a slow boot
  if grep -qE "ValidationError|Traceback|ModuleNotFoundError|ImportError|unrecognized arguments|DEVICE_LOST|OUT_OF_RESOURCES" "$LOG" 2>/dev/null; then
    bad fatal "engine log contains a fatal signature"
  else
    ok fatal
  fi
  # 9. V2 model runner, without which dspark does not exist
  if grep -qiE "Using V2 Model Runner|v2 model runner" "$LOG" 2>/dev/null; then
    ok v2
  else
    bad v2 "V2 model runner not confirmed -- dspark is V2-only"
  fi
  # 10. KV cache dtype, when the profile declares one
  if [[ -n "$kv" ]]; then
    if grep -qiE "cache_dtype.: '$kv|kv[_ ]?cache.{0,20}$kv" "$LOG" 2>/dev/null; then
      ok kv
    else
      bad kv "KV cache dtype $kv not confirmed in the log"
    fi
  fi

  return $(( nbad > 125 ? 125 : nbad ))
}
