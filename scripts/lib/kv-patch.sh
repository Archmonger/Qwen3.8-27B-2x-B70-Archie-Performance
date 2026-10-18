#!/usr/bin/env bash
# KV-offload hotfix attestation, shared by setup.sh and scripts/falsify/kv-patch-cases.sh.
#
# Extracted for the reason attest.sh was: a harness that re-implements the probe
# drifts from it, and a drifting harness keeps passing while the live check is
# broken. Both the boot-time gate and its falsification run THIS code.
#
# Usage:  source scripts/lib/kv-patch.sh
#         kv_probe <image>        -> prints the image's own verdict plus diagnostics
#         kv_classify <raw-text>  -> prints exactly PATCHED | UNPATCHED | UNKNOWN
#         kv_attest <image>       -> sets kv_patch_state / kv_probe_out, prints
#                                    diagnostics on UNKNOWN, returns the state as
#                                    an exit code: 0 patched, 1 otherwise
#
# Why the probe checks the image's bytes instead of image/Dockerfile: setup.sh
# skips the build whenever the tag already exists, so editing the Dockerfile does
# not touch an existing image. The engine then boots, /health answers 200, and the
# offloading-connector AssertionError is waiting for the first long request. Only
# the image's own scheduler.py can attest that the fix is in what actually runs.
#
# Why two exact greps rather than one loose marker: the new bound and the three-way
# min together ARE the fix, so a stray `num_keyed_chunks` from an unrelated future
# edit must not be able to attest a fix that is not there.
#
# Why the verdict must survive diagnostics on stdout: `import vllm` emits a torch
# "XPU device count is zero" warning on any host with no GPU visible to the probe
# container, and that lands on the same stream. So the verdict is one whole line on
# its own and the classifier matches whole lines only -- never a substring.

KV_PATCH_REL="distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
KV_PATCH_MARK_BOUND='num_keyed_chunks = len(group_state.offload_keys)'
KV_PATCH_MARK_MIN='return min(num_chunks, num_allocated_chunks, num_keyed_chunks)'

kv_probe() { # $1=image
  docker run --rm --entrypoint sh "$1" -c '
    V="$(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null | tail -n 1)"
    [ -n "$V" ] || { echo "kv-probe: importing vllm produced no package path" >&2; exit 2; }
    F="$V/'"$KV_PATCH_REL"'"
    [ -f "$F" ] || { echo "kv-probe: missing $F" >&2; exit 2; }
    if grep -q "'"$KV_PATCH_MARK_BOUND"'" "$F" \
       && grep -q "'"$KV_PATCH_MARK_MIN"'" "$F"; then
      echo PATCHED
    else
      echo UNPATCHED
    fi
  ' 2>&1
}

# Whole-line matches only. A diagnostic that merely contains the word PATCHED
# cannot attest the fix, and an empty or unparseable result must not default to
# either verdict: unable to prove the fix is present is not the same as proving
# it is absent, and it is certainly not proof of presence.
kv_classify() { # $1=raw probe output
  # Exactly-one-of, not first-match: output carrying both verdicts means something
  # went wrong upstream, and "the first line happened to say PATCHED" must not be
  # allowed to attest a fix.
  local n_p n_u
  n_p=$(grep -cx "PATCHED" <<<"$1" || true)
  n_u=$(grep -cx "UNPATCHED" <<<"$1" || true)
  if   [[ "$n_p" == "1" && "$n_u" == "0" ]]; then echo PATCHED
  elif [[ "$n_u" == "1" && "$n_p" == "0" ]]; then echo UNPATCHED
  else echo UNKNOWN; fi
}

kv_attest() { # $1=image; sets kv_patch_state and kv_probe_out
  kv_probe_out=$(kv_probe "$1") || true
  kv_patch_state=$(kv_classify "$kv_probe_out")
  [[ "$kv_patch_state" == "PATCHED" ]]
}

# The severity decision, separated from the probe so that the falsification harness
# can exercise the branch table directly. It is the part most likely to be quietly
# wrong: the probe is easy to read and hard to get wrong, while "fatal when the tier
# is on, warning when it is off" is a three-way table over two inputs.
#
# Severity follows the config, because the config is what decides whether the defect
# is reachable at all. With KV_OFFLOAD_SIZE=0 the connector is never constructed
# (config/vllm.py returns early when kv_offloading_size is None), so an unpatched
# image cannot fire the assert and blocking the boot would be theatre. With the tier
# on it is fatal: the crash takes the whole engine down, and only at long context --
# far enough past every pre-deploy gate to be missed.
#
# UNKNOWN is treated like UNPATCHED. Unable to prove the fix is present is not the
# same as proving it is, and the tier-on branch is the one that can crash.
#
# Usage:  kv_severity <patch_state> <kv_offload_size>   -> prints build|fatal|warn|ok
kv_severity() { # $1=patch_state $2=kv_offload_size
  local state="$1" size="${2:-0}"
  if [[ "$state" == "PATCHED" ]]; then
    echo ok
  elif [[ "$size" == "0" || -z "$size" ]]; then
    echo warn
  else
    echo fatal
  fi
}
