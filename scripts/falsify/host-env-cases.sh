#!/usr/bin/env bash
# Falsification harness for load_host_overlay / host_overlay_path in
# scripts/lib/profile-env.sh -- the mechanism that lets ONE machine run the host
# KV tier while the repository ships it off.
#
# Why this exists: an attestation whose verdict paths cannot fail is not an
# attestation. The overlay is precisely the kind of thing that goes wrong
# silently -- if it is read too late it is ignored, if the key whitelist leaks it
# becomes a way to override a measured knob off-book, and if a missing file is
# treated as "no overlay" the box quietly drops to the repo default while every
# check still passes. Each case below asserts BOTH the resolved value and the
# exit status, so a guard that has been emptied out shows up as a wrong verdict
# rather than as a green run.
#
# It sources scripts/lib/profile-env.sh rather than reimplementing the parsing, so
# a harness that drifts from the real code cannot keep passing while the guard is
# broken.
#
# Usage: scripts/falsify/host-env-cases.sh
# Exit 0 only if every case discriminates.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/../lib/profile-env.sh"

# A scratch profile dir. Deliberately NOT inside the repo: the default overlay
# location is resolved relative to the profile file, so writing a host.env next to
# the real profiles/fp8/fp8.env would change what setup.sh does on this box.
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
PROF="$W/fp8.env"

fails=0
ok()   { echo "ok   $*"; }
bad()  { echo "FAIL $*"; fails=$((fails+1)); }

# Write the committed-style profile (always 0, the repo default).
printf 'KV_OFFLOAD_SIZE=0\nIMAGE=someimage:tag\n' > "$PROF"

# Resolve the tier the way the callers do: profile first, then overlay. Prints
# "value|rc" so a case can assert both. The subshell keeps the export from
# leaking between cases -- an earlier export is itself a way to fake a pass.
resolve() {
  (
    unset KV_OFFLOAD_SIZE KV_HOST_ENV 2>/dev/null
    [[ "${1:-}" == "TOK_SET"   ]] && export KV_HOST_ENV="$2"
    [[ "${1:-}" == "TOK_EMPTY" ]] && export KV_HOST_ENV=""
    load_env_file "$PROF" >/dev/null
    load_host_overlay "$PROF" >/dev/null 2>&1
    printf '%s|%s' "${KV_OFFLOAD_SIZE:-UNSET}" "$?"
  )
}
# Same, but keeps stderr so a refusal message can be asserted on content, not just
# on the exit code. A guard that exits 2 for the wrong reason is still a guard
# that can be fooled.
resolve_err() {
  (
    unset KV_OFFLOAD_SIZE KV_HOST_ENV 2>/dev/null
    [[ "${1:-}" == "TOK_SET"   ]] && export KV_HOST_ENV="$2"
    [[ "${1:-}" == "TOK_EMPTY" ]] && export KV_HOST_ENV=""
    load_env_file "$PROF" >/dev/null
    load_host_overlay "$PROF" 2>&1 >/dev/null
  )
}
chk() { # $1=desc $2=got $3=want
  if [[ "$2" == "$3" ]]; then ok "$1 -> $2"; else bad "$1 -> got '$2' want '$3'"; fi
}

# --- the baseline that makes every case below meaningful -------------------
# With no overlay at all the tier must be the committed 0. If this came back 50
# the harness itself would be broken, and every "overlay wins" case would be
# proving nothing.
chk "no overlay -> repo default"  "$(resolve)" "0|0"

# --- the one case that is the whole point of the feature -------------------
printf 'KV_OFFLOAD_SIZE=50\n' > "$W/host.env"
chk "default host.env wins over profile" "$(resolve)" "50|0"

# Explicit path beats the default location.
printf 'KV_OFFLOAD_SIZE=7\n' > "$W/elsewhere.env"
chk "KV_HOST_ENV names another file" "$(resolve TOK_SET "$W/elsewhere.env")" "7|0"

# Explicit disable. Distinct from unset: this is what the falsify harnesses use
# so a developer's own host.env cannot override the values they mutate.
chk "KV_HOST_ENV='' disables the overlay" "$(resolve TOK_EMPTY "")" "0|0"
# The disable must survive an overlay that WOULD have applied.
chk "KV_HOST_ENV='' wins over an existing default" "$(resolve TOK_EMPTY "")" "0|0"

# --- the guards: each must actually refuse --------------------------------
printf 'DTYPE=float16\n' > "$W/host.env"
r="$(resolve)"
chk "overlay may not set DTYPE (value)"  "${r%%|*}" "0"
chk "overlay may not set DTYPE (rc)"     "${r##*|}" "2"
grep -qi 'only set KV_OFFLOAD_SIZE' <<<"$(resolve_err)" \
  || bad "refusal for a foreign key does not say why"

printf 'KV_OFFLOAD_SIZE=abc\n' > "$W/host.env"
r="$(resolve)"
chk "non-numeric rejected (value)" "${r%%|*}" "0"
chk "non-numeric rejected (rc)"    "${r##*|}" "2"
grep -qi 'not a number' <<<"$(resolve_err)" \
  || bad "refusal for a non-numeric value does not say why"

# A decimal is legitimate: vLLM takes a float, and serve.sh's guard is != "0".
printf 'KV_OFFLOAD_SIZE=12.5\n' > "$W/host.env"
chk "decimal accepted" "$(resolve)" "12.5|0"

# An explicitly named file that does not exist is a mistake, not an absence.
# Silently returning "no overlay" here is the inert-knob failure: the operator
# believes the host value is in force while the box runs the repo default.
r="$(resolve TOK_SET "$W/definitely-absent.env")"
chk "explicit missing overlay (value)" "${r%%|*}" "0"
chk "explicit missing overlay (rc)"    "${r##*|}" "2"
grep -qi 'was set but does not' <<<"$(resolve_err TOK_SET "$W/definitely-absent.env")" \
  || bad "refusal for a missing explicit overlay does not name the file"

# Comments, blanks and CRLF must not upset the single-key rule.
printf '# comment\r\n\r\nKV_OFFLOAD_SIZE=25\r\n' > "$W/host.env"
chk "CRLF + comments parse" "$(resolve)" "25|0"

# Quoting, matching load_env_file's own tolerance.
printf 'KV_OFFLOAD_SIZE="33"\n' > "$W/host.env"
chk "quoted value unquoted" "$(resolve)" "33|0"

# --- host_overlay_path: what setup.sh hands to compose ---------------------
# Compose hard-fails on a --env-file that does not exist, so the path helper must
# print NOTHING rather than a path when there is no overlay. A helper that
# echoed the default path unconditionally would break every run on a box without
# one -- which is every box but ours.
printf 'KV_OFFLOAD_SIZE=50\n' > "$W/host.env"
p="$(host_overlay_path "$PROF" 2>/dev/null)"; rc=$?
chk "path helper prints an existing overlay" "$([[ -n "$p" ]] && echo yes)" "yes"
chk "path helper rc when present"            "$rc" "0"
rm -f "$W/host.env"
p="$(host_overlay_path "$PROF" 2>/dev/null)"; rc=$?
chk "path helper prints nothing when absent" "${p:-EMPTY}" "EMPTY"
chk "path helper rc when absent"             "$rc" "1"

echo
if [[ $fails -eq 0 ]]; then
  echo "falsify/host-env: PASS -- the host overlay can only ever move KV_OFFLOAD_SIZE,"
  echo "                    and it refuses rather than silently doing nothing"
else
  echo "falsify/host-env: FAIL ($fails) -- do not trust the host overlay until this is green"
  exit 1
fi
