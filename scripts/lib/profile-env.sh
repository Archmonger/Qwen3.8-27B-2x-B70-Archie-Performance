#!/usr/bin/env bash
# Shared profile-env loader, sourced by setup.sh, verify.sh, soak.sh and
# scripts/bench/run-all.sh.
#
# Why this is a parser and not `set -a; . profile.env`:
#
#   EXTRA_VLLM_ARGS=--compilation-config {"mode":"none","cudagraph_mode":...}
#
# Sourcing that line hands bash brace expansion a command word and fails with
# `mode:none: command not found` -- and under a script that continues after the
# error, EXTRA_VLLM_ARGS comes back empty, so the engine boots with no graphs and
# no ceiling at all. Compose reads the same file with its own parser and passes
# the value through verbatim, so this one has to as well.
#
# Deliberately minimal: KEY=value lines, whole-line comments, CRLF tolerance, and
# one matched pair of surrounding quotes stripped. No command substitution, no
# expansion of $VAR inside values -- a profile file must not be able to execute
# anything.

load_env_file() { # $1=path
  local _lef_line _lef_k _lef_v
  while IFS= read -r _lef_line || [[ -n "$_lef_line" ]]; do
    _lef_line="${_lef_line%$'\r'}"
    [[ "$_lef_line" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "$_lef_line" == *=* ]] || continue
    _lef_k="${_lef_line%%=*}"
    _lef_v="${_lef_line#*=}"
    # Trim surrounding whitespace from the key only: compose does not trim values,
    # and a trailing space inside EXTRA_VLLM_ARGS would change the argv it produces.
    _lef_k="${_lef_k#"${_lef_k%%[![:space:]]*}"}"
    _lef_k="${_lef_k%"${_lef_k##*[![:space:]]}"}"
    [[ "$_lef_k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    case "$_lef_k" in _lef_*) continue ;; esac
    case "$_lef_v" in
      \"*\") [[ ${#_lef_v} -ge 2 ]] && _lef_v="${_lef_v:1:${#_lef_v}-2}" ;;
      \'*\') [[ ${#_lef_v} -ge 2 ]] && _lef_v="${_lef_v:1:${#_lef_v}-2}" ;;
    esac
    export "$_lef_k=$_lef_v"
  done < "$1"
}

# ---------------------------------------------------------------------------
# Host-only override, for the one knob that must differ from the repo default.
#
# WHY THIS EXISTS. Everything committed to the repository ships KV_OFFLOAD_SIZE=0:
# the tier is a crash vector at long context (docs/kv-offload-defect.md) and has
# never served a read on this workload, so nobody adopting this recipe should
# inherit it switched on. One specific box does want the capacity, and the honest
# place for that is a file git cannot see. The alternative -- keeping 50 in
# profiles/fp8/fp8.env as an uncommitted local edit, which is where it lived until
# now -- is worse twice over: the repo carries a value it does not endorse, and
# `git checkout`/`git pull` silently reverts it to 0 while every other signal stays
# green. That is a config that changes underneath you.
#
# WHY EXACTLY ONE KEY. A general "last file wins" overlay is how a host drifts on
# GPU_MEM_UTIL, DTYPE, or the graph ceiling while the repo still claims the
# measured configuration -- this project has been bitten by inert and unasserted
# knobs repeatedly. Narrowing the overlay to the single knob that genuinely needs
# to be host-specific keeps every other value single-sourced from the committed
# profile. A second key should be added here deliberately, one at a time, each with
# its own attestation rule.
#
# KV_HOST_ENV overrides the path; setting it to the EMPTY STRING disables the
# overlay outright. That distinction is load-bearing and not a convenience: an
# unset variable means "use the default location", while an empty one means "do
# not read any overlay". scripts/falsify/kv-patch-cases.sh mutates fp8.env to
# drive the setup.sh tier gate, and a developer's host.env pinned at 50 would
# silently override the values those cases set -- turning a tier-off case into a
# tier-on case that still reports PASS. The harness sets KV_HOST_ENV="" to opt out.

# Single definition of the overlay's location, printed only when a usable file
# exists. setup.sh needs the same path to pass as compose's final --env-file; a
# second copy of this rule is how the two would drift into attesting one file and
# baking another. Compose hard-fails on a missing --env-file, so callers must
# check this rather than assume the default path exists.
host_overlay_path() { # $1=profile env file -> existing path, or nothing
  # rc 1 = no overlay applies (default location absent, or explicitly disabled)
  # rc 2 = KV_HOST_ENV names a file that does not exist. That is a configuration
  # mistake, not an absence: the operator asked for an overlay they did not
  # create, and reading no file would leave the tier at the repo default while
  # every log still implies the host value is in force -- the inert-knob failure
  # this project keeps meeting. Fail loudly instead.
  if [[ -n "${KV_HOST_ENV+set}" ]]; then      # set at all, even to ""
    [[ -n "$KV_HOST_ENV" ]] || return 1       # explicitly disabled
    [[ -f "$KV_HOST_ENV" ]] || return 2
    printf '%s\n' "$KV_HOST_ENV"
  else
    local d
    d="$(dirname "$1")/host.env"
    [[ -f "$d" ]] || return 1
    printf '%s\n' "$d"
  fi
}

load_host_overlay() { # $1=profile env file; returns 2 on a bad/missing explicit overlay
  local f k _ho_line _ho_val rc=0
  f="$(host_overlay_path "$1")" || rc=$?
  case $rc in
    0) ;;
    1) return 0 ;;                            # no overlay; the normal case
    *) echo "[profile-env] REFUSING: KV_HOST_ENV=$KV_HOST_ENV was set but does not" >&2
       echo "              exist. Unset it to use the repo default, or create the" >&2
       echo "              file. Silently ignoring it would leave the KV tier at 0" >&2
       echo "              while you believe the host value is in force." >&2
       return 2 ;;
  esac
  while IFS= read -r _ho_line || [[ -n "$_ho_line" ]]; do
    _ho_line="${_ho_line%$'\r'}"
    [[ "$_ho_line" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "$_ho_line" == *=* ]] || continue
    k="${_ho_line%%=*}"; _ho_val="${_ho_line#*=}"
    k="${k#"${k%%[![:space:]]*}"}"; k="${k%"${k##*[![:space:]]}"}"
    case "$_ho_val" in
      \"*\") [[ ${#_ho_val} -ge 2 ]] && _ho_val="${_ho_val:1:${#_ho_val}-2}" ;;
      \'*\') [[ ${#_ho_val} -ge 2 ]] && _ho_val="${_ho_val:1:${#_ho_val}-2}" ;;
    esac
    if [[ "$k" != "KV_OFFLOAD_SIZE" ]]; then
      echo "[profile-env] REFUSING host overlay $f: may only set KV_OFFLOAD_SIZE," >&2
      echo "              got '$k'. Every other knob stays single-sourced from the" >&2
      echo "              committed profile -- that restriction is the point." >&2
      return 2
    fi
    # Numeric or nothing. A typo'd value would otherwise reach serve.sh, whose
    # guard is a literal != "0" test, and hand vLLM --kv-offloading-size abc.
    if ! [[ "$_ho_val" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      echo "[profile-env] REFUSING host overlay $f: KV_OFFLOAD_SIZE='$_ho_val'" >&2
      echo "              is not a number." >&2
      return 2
    fi
    export KV_OFFLOAD_SIZE="$_ho_val"
    echo "[profile-env] host overlay $f -> KV_OFFLOAD_SIZE=$_ho_val"
  done < "$f"
  return 0
}
