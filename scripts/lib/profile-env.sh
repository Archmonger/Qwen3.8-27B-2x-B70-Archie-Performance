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
