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
#         boot_slice <timestamps-log> <boot-start-iso> <out-file>
#           -> keeps only lines from the current boot; see boot_slice for why.
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

# Slice a `docker logs --timestamps` log down to one boot.
#
# WHY THIS EXISTS. `docker logs` returns every boot the container has ever had, and
# the fatal-signature rule scans whatever log it is handed. That made attestation
# permanently red on a container that had once crashed: a real UR_RESULT_ERROR_DEVICE_LOST
# killed one boot, the supervisor restarted it, and verify.sh then refused to attest a
# healthy engine by quoting the dead boot's crash. The rule's question is "is THIS boot
# a dead engine pretending to be slow", so it must only see THIS boot.
#
# $1 = timestamps log   $2 = ISO-8601 boot start   $3 = output file
# Prints the number of lines kept. Returns 1 on an empty slice so the caller cannot
# mistake "no log for this window" for "a log with nothing wrong in it".
#
# Deliberately FAIL-CLOSED: with no usable boot start it copies the WHOLE log, which
# can only ever produce more fatal-signature hits, never fewer. A helper that silently
# narrowed the window on a parse failure would turn a missing timestamp into a pass.
boot_slice() {
  local src="$1" since="$2" out="$3"
  [[ -s "$src" ]] || { : > "$out"; return 1; }
  if [[ -z "$since" ]] || [[ ${#since} -lt 16 ]]; then
    cat "$src" > "$out"
    wc -l < "$out" | tr -d ' '
    return 0
  fi
  # Compare the leading RFC3339 timestamp lexically: identical format and zero-padded,
  # so lexical order IS chronological order and no date parsing or locale is involved.
  # The 19-char prefix is to the second; docker's StartedAt carries nanoseconds, which
  # would otherwise exclude the first log lines printed in that same second.
  awk -v since="${since:0:19}" '
    { ts = substr($1, 1, 19)
      if (ts == "" || ts >= since) print }' "$src" > "$out"
  local kept; kept=$(wc -l < "$out" | tr -d ' ')
  [[ "$kept" -gt 0 ]] || return 1
  echo "$kept"
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
  #
  # These helpers are NAMESPACED on purpose. They used to be plain ok()/bad(), and
  # because bash functions are global, running attest() overwrote the caller's own
  # ok()/bad() for the rest of the script: verify.sh defines bad() to print "  FAIL"
  # and increment `fails`, and after attest() clobbered it, every later check printed
  # a stray "rule|..." line and `fails` stopped counting -- a check that can no longer
  # make the script exit non-zero. Namespacing is what makes that impossible rather
  # than merely unlikely. It surfaced as "attest.sh: line 53: nbad: unbound variable",
  # which is the caller's bad() reaching for attest()'s long-gone local.
  _attest_ok()  { echo "rule|$1|ok|"; }
  _attest_bad() { echo "rule|$1|bad|${2:-}"; nbad=$((nbad+1)); }

  # 1. graph ceiling equals MAX_NUM_SEQS x (1+k), as computed by the caller
  [[ "$ceil" == "$want_ceil" ]] && _attest_ok ceil || _attest_bad ceil "max_cudagraph_capture_size=${ceil:-missing}, expected $want_ceil"
  # 2. the capture ladder actually contains the saturated batch size
  [[ "$maxsize" == "$want_ceil" ]] && _attest_ok ladder || _attest_bad ladder "capture ladder tops out at ${maxsize:-none}, must contain $want_ceil ($sizes)"
  # 3. graphs on, and the full-decode kind
  [[ "$cg" == *"FULL_DECODE_ONLY"* ]] && _attest_ok cg || _attest_bad cg "cudagraph_mode=${cg:-missing}"
  # 4. static k equals the measured value (skipped when no k is expected)
  if [[ -n "$want_k" ]]; then
    [[ "$nsp" == "$want_k" ]] && _attest_ok k || _attest_bad k "num_speculative_tokens=${nsp:-missing}, expected $want_k"
  fi
  # 5. dspark, not dflash (dflash does not boot with the published drafter on 0.26)
  [[ "$meth" == "dspark" ]] && _attest_ok meth || _attest_bad meth "speculative method=${meth:-missing}"
  # 6. no dynamic schedule: with graphs on it garbles output at healthy counters
  [[ -z "$sched" ]] && _attest_ok sched || _attest_bad sched "dynamic schedule active: $sched"
  # 7. capture actually ran -- absent means every shape is eager
  [[ -n "$cap" ]] && _attest_ok capture || _attest_bad capture "no 'Graph capturing finished' line -- nothing was graphed"
  # 8. no fatal signature masquerading as a slow boot
  if grep -qE "ValidationError|Traceback|ModuleNotFoundError|ImportError|unrecognized arguments|DEVICE_LOST|OUT_OF_RESOURCES" "$LOG" 2>/dev/null; then
    _attest_bad fatal "engine log contains a fatal signature"
  else
    _attest_ok fatal
  fi
  # 9. V2 model runner, without which dspark does not exist
  if grep -qiE "Using V2 Model Runner|v2 model runner" "$LOG" 2>/dev/null; then
    _attest_ok v2
  else
    _attest_bad v2 "V2 model runner not confirmed -- dspark is V2-only"
  fi
  # 10. KV cache dtype, when the profile declares one
  if [[ -n "$kv" ]]; then
    if grep -qiE "cache_dtype.: '$kv|kv[_ ]?cache.{0,20}$kv" "$LOG" 2>/dev/null; then
      _attest_ok kv
    else
      _attest_bad kv "KV cache dtype $kv not confirmed in the log"
    fi
  fi

  return $(( nbad > 125 ? 125 : nbad ))
}
