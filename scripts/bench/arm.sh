#!/usr/bin/env bash
# One arm = one boot of the serving image with caller-supplied overrides, then
# measure, gate, attest, tear down. For A/B work where the knob is an
# environment variable or a serve flag.
#
# Why this exists instead of "edit the compose file and restart": A/B work on
# this host keeps getting decided by drift rather than by the knob. The box moves
# ~5% on the natural probe per 10 boots, and two earlier decisions were nearly
# settled by comparing an arm against a control measured in a different session
# under different harness code. So this file enforces three things:
#
#   1. ATTESTATION BEFORE MEASUREMENT. The ceiling, cudagraph mode and capture
#      ladder are read back from the engine log via scripts/lib/attest.sh and
#      must match the profile, or the arm aborts. An arm that silently reverted
#      to eager would otherwise be scored as "the knob did nothing" when it never
#      applied -- that confound once hid a real 38% batched win across two
#      deploys.
#   2. A GATE, NOT JUST A NUMBER. Throughput alone cannot be trusted: the
#      graph-stride defect runs at full speed with healthy acceptance counters
#      while producing corrupt text. correctness.sh and acceptance.sh must pass
#      or the row records gate=FAIL next to the numbers, so a fast-but-wrong arm
#      can never be quoted without its failure visible.
#   3. AN ENGAGEMENT ASSERTION. An inert knob and a neutral knob look identical
#      in a throughput table. NEEDLE asserts a regex appears in the engine log
#      (or in the dispatch census when PROBES= is given); NEEDLE_ABSENT asserts
#      the converse. Without it you measure nothing and report "no effect".
#
# Every measurement is delegated to the harness that produced the recorded
# table, so an arm is directly comparable to a recorded cell:
#   correctness  scripts/bench/correctness.sh   (PORT/SERVED_NAME/REPS env)
#   acceptance   scripts/bench/acceptance.sh    (PORT/SERVED_NAME env)
#   throughput   scripts/bench/conc.py          (canonical budget, no override)
#
# Production owns both GPUs, so stop it first: two engines on one pair both
# mis-size the KV pool and fail with "Free memory on device xpu:N ... less than
# desired", and numbers taken that way mean nothing.
#
# Usage:
#   scripts/bench/arm.sh LABEL PORT [KEY=VALUE ...]
#
# Recognised (non-passthrough) keys:
#   DTYPE=float16               append --dtype <v>
#   MAMBA_SSM=float16           append --mamba-ssm-cache-dtype <v>
#   KVDT=fp8|auto|...           override the profile's KV_CACHE_DTYPE
#   NEEDLE=regex                engine log/census MUST match (engagement proof)
#   NEEDLE_ABSENT=regex         engine log/census must NOT match
#   PROBES=dir                  bind-mount dir/{layernorm.py,flash_attn.py,
#                               _xpu_esimd.py,xpu.py} over the image's copies for
#                               dispatch-census instrumentation (optional).
#                               See docs/measurement-notes.md, "Prove the knob
#                               engaged" -- a census note must sit at the op()
#                               call site, not at function entry.
# Anything else is exported into the container, so any profile knob
# (GPU_MEM_UTIL, MAX_NUM_SEQS, ...) can be varied per arm.
#
# W57_HOLD=1 leaves the container running after the row is recorded so an
# external battery (quality.py / gsm8k_pair.py / suite.py) can be aimed at the
# exact boot that produced the numbers. The row is written BEFORE the hold.
#
# ARM_CHECK_ONLY=1 validates the rig (profile, image, device nodes, filtered
# by-path dir, the harnesses this script delegates to) and exits without
# touching docker. Run it BEFORE stopping production: every rig error here
# otherwise surfaces as "BOOT FAILED" after the GPUs were already handed over,
# and reads like the knob broke the engine.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/../lib/profile-env.sh"
. "$HERE/../lib/attest.sh"
. "$HERE/../lib/degenerate.sh"

LABEL=${1:?label: scripts/bench/arm.sh LABEL PORT [KEY=VALUE ...]}
# Deliberately NOT named PORT. load_env_file below sources the profile, and
# profiles/fp8/fp8.env sets PORT=8103 -- sourcing it after the positional parse
# overwrote the arm's port, so every arm booted on the profile port, collided
# with production on 8103, and produced "Free memory on device" failures that
# looked like a GPU problem. Caught by running this script against stubbed
# docker/curl: the boot loop reported port 8103 for an arm invoked with 8199.
ARM_PORT=${2:?port}
shift 2

PROFILE=${PROFILE:-fp8}
PROFILE_ENV="$HERE/../../profiles/$PROFILE/$PROFILE.env"
[[ -f "$PROFILE_ENV" ]] || { echo "arm: no such profile: $PROFILE" >&2; exit 2; }
# Absolute, because fp8.env sets RUNSTATE_DIR=./runstate and compose resolves
# that against the compose file's own directory. Here the CWD is whoever invoked
# the script, so a bare ./runstate would mount the wrong (or no) filtered
# by-path dir -- and oneCCL aborts in fill_device_fds when that dir is wrong,
# which looks nothing like a path bug.
PROFILE_DIR="$(cd "$(dirname "$PROFILE_ENV")" && pwd)"
# Snapshot before sourcing: profiles/fp8/fp8.env sets IMAGE, CEILING,
# GPU_MEM_UTIL, MAX_NUM_SEQS, MODELS_DIR, TARGET_DIR, DRAFTER_DIR, CACHE_DIR and
# PORT, and load_env_file exports every one of them unconditionally. Sourcing it
# over any of those silently reverts the caller's override, which turns an A/B
# into two identical arms -- an inert knob reporting "no effect". So the caller's
# value always wins: whatever was already in the environment, or in the KEY=VAL
# arguments, is re-exported after the profile is read.
# Plain K=V list, not an associative array: ${!ARR[@]} inside a ${...+...}
# guard parses as indirect expansion, so the guard form aborted with
# "invalid variable name" and silently applied none of the overrides.
CALLER_OVERRIDES=()
for _k in IMAGE CEILING GPU_MEM_UTIL MAX_NUM_SEQS MAX_MODEL_LEN MODELS_DIR \
          TARGET_DIR DRAFTER_DIR CACHE_DIR PORT NUM_SPEC_TOKENS KV_CACHE_DTYPE \
          SPEC COMPILE XPU_GRAPH PREFIX_CACHE DTYPE MAMBA_SSM_CACHE_DTYPE \
          EXTRA_VLLM_ARGS RUNSTATE_DIR; do
  [[ -n "${!_k:-}" ]] && CALLER_OVERRIDES+=("$_k=${!_k}")
done
# Positional KEY=VAL beats the environment, which beats the profile.
for _kv in "$@"; do
  _k=${_kv%%=*}
  [[ "$_k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && [[ "$_kv" == *=* && -n "${_kv#*=}" ]] || continue
  for _i in "${!CALLER_OVERRIDES[@]}"; do
    [[ "${CALLER_OVERRIDES[$_i]%%=*}" == "$_k" ]] && unset 'CALLER_OVERRIDES[_i]'
  done
  CALLER_OVERRIDES+=("$_kv")
done
unset _k _kv _i

load_env_file "$PROFILE_ENV"

# Re-apply the caller's overrides on top of the profile.
for _kv in ${CALLER_OVERRIDES[@]+"${CALLER_OVERRIDES[@]}"}; do export "$_kv"; done
unset _kv
# fp8.env carries RUNSTATE_DIR=./runstate. Resolve it to an absolute path here
# rather than expanding it inside the docker(1) argv: it is unquoted there, so a
# value with a space would word-split into a second -v argument. A relative
# value is resolved against the profile directory, which is what compose does;
# an absolute one is used as given.
case "${RUNSTATE_DIR:-./runstate}" in
  ./*) DRI_DIR="$PROFILE_DIR/${RUNSTATE_DIR#./}" ;;
  /*)  DRI_DIR="$RUNSTATE_DIR" ;;
  *)   DRI_DIR="$PROFILE_DIR/$RUNSTATE_DIR" ;;
esac
[[ -d "$DRI_DIR/dri-by-path" ]] || {
  echo "arm: no filtered by-path dir at $DRI_DIR/dri-by-path -- run setup.sh" >&2
  echo "     oneCCL aborts in fill_device_fds on an unfiltered /dev/dri/by-path." >&2
  exit 2; }
DEVICES_ENV=${DEVICES_ENV:-$PROFILE_DIR/runstate/devices.env}
[[ -f "$DEVICES_ENV" ]] || {
  echo "arm: missing $DEVICES_ENV -- run setup.sh (or pass DEVICES_ENV=...)." >&2
  echo "     Without it the --device args render empty and the container sees no GPU." >&2
  exit 2; }
load_env_file "$DEVICES_ENV"

O=${ARM_OUT_DIR:-$HERE/../../arm-runs}
mkdir -p "$O"
ROWS=${ARM_ROWS:-$O/arm_rows.jsonl}
LOGF=$O/arm.log
IMAGE=${IMAGE:-vllm-llmscaler:icr26.35}
CTR="vllm-arm-$(echo "$LABEL" | tr 'A-Z' 'a-z')"
MNAME="arm-$(echo "$LABEL" | tr 'A-Z' 'a-z')"
CEILING=${CEILING:-80}
# No defaults here. These three come from the profile env, and inventing a
# fallback would mean silently bind-mounting whatever directory happened to exist
# on the machine running the arm -- a wrong checkpoint would then be measured and
# recorded as the profile's, with plausible numbers. Requiring them is what makes
# a misconfigured rig fail before the GPUs are touched rather than after.
for _v in MODELS_DIR TARGET_DIR DRAFTER_DIR; do
  [[ -n "${!_v:-}" ]] || { echo "arm: $_v is unset and $PROFILE_ENV did not provide it -- refusing to guess a model path" >&2; exit 2; }
done
MODELS=$MODELS_DIR
TARGET_SRC=$TARGET_DIR
DRAFTER_SRC=$DRAFTER_DIR

# Not `DTYPE=`: the caller may have set DTYPE in the environment (it is in the
# override list above), and clearing it here would drop it -- an fp16 arm would
# silently boot at the profile's bfloat16 and be recorded as an fp16 result.
DTYPE=${DTYPE:-}; MAMBA_SSM=""; KVDT=${KV_CACHE_DTYPE:-fp8}
NEEDLE=""; NEEDLE_ABSENT=""; PROBES=""; PASS_ENV=()
for kv in "$@"; do
  case "$kv" in
    DTYPE=*)         DTYPE=${kv#*=} ;;
    MAMBA_SSM=*)     MAMBA_SSM=${kv#*=} ;;
    KVDT=*)          KVDT=${kv#*=} ;;
    NEEDLE=*)        NEEDLE=${kv#*=} ;;
    NEEDLE_ABSENT=*) NEEDLE_ABSENT=${kv#*=} ;;
    PROBES=*)        PROBES=${kv#*=} ;;
    *)               PASS_ENV+=("$kv") ;;
  esac
done
# The dtype knobs reach the launcher as the env vars scripts/serve.sh already
# understands, NOT as extra argv: a second route to --dtype could emit the flag
# twice, and the env route is the one that ships, so exercising it is the point.
[[ -n "$DTYPE" ]] && export DTYPE
[[ -n "$MAMBA_SSM" ]] && export MAMBA_SSM_CACHE_DTYPE="$MAMBA_SSM"

say(){ echo "[$(date -u +%H:%M:%S)] [$LABEL] $*" | tee -a "$LOGF"; }
row(){ # $1 = failure tag; anything after it is recorded as the arm's env.
  # Not "$@": that re-passes $1 and duplicates the tag into the env field.
  python3 -c "import json,sys;print(json.dumps({'label':sys.argv[1],'port':int(sys.argv[2]),'fail':sys.argv[3],'env':' '.join(sys.argv[4:])}))" \
    "$LABEL" "$ARM_PORT" "$1" "${*:2}" >> "$ROWS"; }

CEIL_JSON="{\"mode\":\"none\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"max_cudagraph_capture_size\":$CEILING}"

ENVARGS=()
for kv in ${PASS_ENV[@]+"${PASS_ENV[@]}"}; do ENVARGS+=(-e "$kv"); done
# Exactly the two B70 nodes and nothing else, as the profile's compose file
# does. The A310 and the RTX 2000 Ada stay OUT: oneCCL aborts in
# fill_device_fds on the dangling by-path entries they bring with them.
GPU_NODES=(
  "/dev/dri/${RANK0_CARD:?}"
  "/dev/dri/${RANK0_RENDER:?}"
  "/dev/dri/${RANK1_CARD:?}"
  "/dev/dri/${RANK1_RENDER:?}"
)
DEVOPTS=()
# Checked before docker(1) sees them: a bad node surfaces as
# "error gathering device information ... not a directory", which reads like a
# permissions or driver fault. A missing/stale node is a config bug -- and when
# the caller already stopped production to hand over the GPUs, a misleading
# BOOT FAILED costs a downtime window. seq6 lost one to exactly this.
for _d in "${GPU_NODES[@]}"; do
  [[ -e "$_d" ]] || { echo "arm: device $_d does not exist on this host" >&2; exit 2; }
  DEVOPTS+=(--device "$_d")
done
# Same reasoning for the harnesses this script delegates to: a missing one
# aborts mid-run, after boot, looking like an engine fault.
for _h in "$HERE/../serve.sh" "$HERE/correctness.sh" "$HERE/acceptance.sh" \
          "$HERE/conc.py" "$HERE/probe.py"; do
  [[ -f "$_h" ]] || { echo "arm: missing $_h" >&2; exit 2; }
done
MOUNTS=()
# Default is the UNMODIFIED image, so an arm stays comparable with the recorded
# table. Census work is opt-in with PROBES=instrument (the modules shipped in
# scripts/instrument/) or PROBES=<dir>. A directory must carry SHA256SUMS: three
# of the four are purely additive logging, but xpu.py also makes the fp8 dispatch
# thresholds env-overridable, so a stale or half-edited copy can change which
# kernel runs while still printing plausible numbers. An unverified probe dir
# once made the GDN core look engaged when it was not.
#
# A GDN census NEEDLE needs SPEC=1 (the default here): try_esimd_gdn_core_spec_decode
# returns early unless metadata.num_spec_decodes == 1 and spec_sequence_masks is
# set, so with spec decode off that path can never be entered and the needle can
# never match. Asking for it under SPEC=0 would report "knob inert" for an
# unrelated reason.
if [[ "${PROBES:-}" == instrument ]]; then
  PROBES="$HERE/../instrument"
fi
if [[ -z "${PROBES:-}" || "${PROBES:-none}" == none ]]; then
  PROBES=""
else
  [[ -f "$PROBES/SHA256SUMS" ]] || {
    echo "arm: $PROBES has no SHA256SUMS -- refusing to mount unverified modules" >&2
    echo "     A stale or half-edited probe dir produced a false kernel claim once:" >&2
    echo "     the entry-site census note looked like proof the GDN core ran." >&2
    exit 2; }
  ( cd "$PROBES" && sha256sum -c SHA256SUMS >/dev/null 2>&1 ) || {
    echo "arm: instrumented modules in $PROBES do not match SHA256SUMS" >&2
    echo "     Fix or re-copy them; do not measure with unverified patches." >&2
    exit 2; }
  for f in layernorm.py flash_attn.py _xpu_esimd.py xpu.py; do
    [[ -f "$PROBES/$f" ]] || { echo "arm: PROBES=$PROBES missing $f" >&2; exit 2; }
    grep -q w57-census "$PROBES/$f" \
      || { echo "arm: $PROBES/$f carries no census probe" >&2; exit 2; }
    # Compile them here: a SyntaxError in a mounted module otherwise surfaces as
    # a ModuleNotFoundError deep in a 400-second boot.
    python3 -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" "$PROBES/$f" \
      || { echo "arm: $PROBES/$f does not parse" >&2; exit 2; }
  done
  MOUNTS=(
    -v "$PROBES/layernorm.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py:ro"
    -v "$PROBES/flash_attn.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py:ro"
    -v "$PROBES/_xpu_esimd.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/_xpu_esimd.py:ro"
    -v "$PROBES/xpu.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/scaled_mm/xpu.py:ro"
  )
fi

# Everything that can be wrong with the rig has now been checked -- profile,
# devices, by-path dir, delegated harnesses, and the probe modules and their
# manifest -- so this is the right place to stop for a preflight. It sits AFTER
# probe validation on purpose: an earlier revision exited here before the probe
# block, so ARM_CHECK_ONLY never validated the modules it was about to mount and
# a bad PROBES= directory only failed after the caller had stopped production.
if [[ "${ARM_CHECK_ONLY:-0}" == "1" ]]; then
  echo "arm: rig OK -- profile=$PROFILE image=$IMAGE port=$ARM_PORT"
  echo "     devices: ${GPU_NODES[*]}"
  echo "     by-path: $DRI_DIR/dri-by-path"
  echo "     dtype=${DTYPE:-profile} ssm=${MAMBA_SSM:-profile} kv=$KVDT ceiling=$CEILING"
  echo "     probes: ${PROBES:-none (unmodified image)}"
  echo "     output: $O"
  exit 0
fi

say "launch port=$ARM_PORT profile=$PROFILE dtype=${DTYPE:-profile} ssm=${MAMBA_SSM:-profile} kv=$KVDT ${PASS_ENV[*]:-}"
docker rm -f "$CTR" >/dev/null 2>&1
sleep 3

timeout "${ARM_BOOT_TIMEOUT:-2700}" docker run -d --name "$CTR" \
  --network host --shm-size 32g --ipc=host \
  "${DEVOPTS[@]}" \
  -v "$DRI_DIR/dri-by-path:/dev/dri/by-path:ro" \
  "${MOUNTS[@]+"${MOUNTS[@]}"}" \
  -v "$MODELS/$TARGET_SRC:/models/target:ro" \
  -v "$MODELS/$DRAFTER_SRC:/models/drafter:ro" \
  -v "$HERE/../serve.sh:/serve.sh:ro" \
  -e VLLM_CACHE_ROOT="$CACHE_DIR/vllm" \
  -e TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor" \
  -e TRITON_CACHE_DIR="$CACHE_DIR/triton" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e SERVED_NAME="$MNAME" -e PORT="$ARM_PORT" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}" -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}" \
  -e GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}" \
  -e PREFIX_CACHE="${PREFIX_CACHE:-1}" -e COMPILE="${COMPILE:-1}" \
  -e XPU_GRAPH="${XPU_GRAPH:-1}" -e SPEC="${SPEC:-1}" \
  -e NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}" -e KV_CACHE_DTYPE="$KVDT" \
  -e CCL_SYCL_ALLREDUCE_TMP_BUF="${CCL_SYCL_ALLREDUCE_TMP_BUF:-1}" \
  -e DTYPE="${DTYPE:-auto}" -e MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-auto}" \
  -e EXTRA_VLLM_ARGS="--compilation-config $CEIL_JSON" \
  "${ENVARGS[@]}" \
  --entrypoint /bin/bash "$IMAGE" /serve.sh >>"$LOGF" 2>&1

L=$O/${LABEL}.engine.log
RAW=$O/${LABEL}.raw.log
snap(){
  # Timestamped raw copy, then boot-sliced into $L the way verify.sh does it:
  # the fatal-signature rule scans whatever log it is handed, so an unslliced log
  # from a container that once crashed stays red forever and would abort good
  # arms over a fault from an earlier boot.
  docker logs --timestamps "$CTR" > "$RAW" 2>&1 || true
  local started
  started=$(docker inspect --format '{{.State.StartedAt}}' "$CTR" 2>/dev/null || true)
  boot_slice "$RAW" "$started" "$L" >/dev/null 2>&1 || cp "$RAW" "$L"
}
trap 'snap; docker rm -f "$CTR" >/dev/null 2>&1 || true' EXIT

alive(){ timeout 10 curl -sf -m 8 "http://127.0.0.1:$ARM_PORT/health" >/dev/null 2>&1; }
# The fatal signature, spelled exactly. NOT "Free memory on device": gpu_worker
# prints that same phrase on an INFO line at every successful startup
# ("Free memory on device (29.18/30.3 GiB) on startup"), so matching the phrase
# aborts a perfectly healthy boot at ~250s. The failure this is meant to catch
# -- a second engine holding the other GPU -- reads
# "Free memory on device xpu:0 (0.82/31.89 GiB) on startup is less than desired".
FATAL='ValidationError|Traceback|ModuleNotFoundError|unrecognized arguments|DEVICE_LOST|OUT_OF_RESOURCES|is less than desired'
ok=0
for j in $(seq 1 500); do
  sleep 3
  if alive; then ok=1; break; fi
  docker logs "$CTR" 2>&1 | grep -qE "$FATAL" && break
  docker ps --format '{{.Names}}' | grep -q "^${CTR}$" || break
done
snap
if [[ $ok -ne 1 ]]; then
  say "BOOT FAILED +$((j*3))s"
  grep -E "$FATAL" "$L" 2>/dev/null | tail -6 | sed 's/^/    /' | cut -c1-200 | tee -a "$LOGF"
  row boot-fail; exit 1
fi
# Identity, not liveness: a /health 200 does not say WHICH engine answered, and a
# stale listener on the arm's port has produced green arms that measured a
# different container.
timeout 20 curl -s "http://127.0.0.1:$ARM_PORT/v1/models" | grep -q "\"id\":\"$MNAME\"" \
  || { say "IDENTITY MISMATCH: $MNAME not served on $ARM_PORT"; row identity; exit 1; }

# --- probe-mount proof -------------------------------------------------------
# The census patches only add logging, so a failed mount still yields valid
# tok/s -- but it also yields no census lines, and "no census lines" is exactly
# the evidence used to declare a path dead. Assert the live files ARE patched.
if [[ -n "$PROBES" ]]; then
  MOUNT_ERR=""
  for spec in \
    "model_executor/layers/layernorm.py" \
    "v1/attention/backends/flash_attn.py" \
    "model_executor/layers/mamba/gdn/_xpu_esimd.py" \
    "model_executor/kernels/linear/scaled_mm/xpu.py"; do
    hits=$(docker exec "$CTR" bash -lc "grep -c w57-census /opt/venv/lib/python3.12/site-packages/vllm/$spec 2>/dev/null" || echo 0)
    [[ "${hits:-0}" -ge 1 ]] || MOUNT_ERR="$MOUNT_ERR unpatched:$spec"
  done
  [[ -z "$MOUNT_ERR" ]] || { say "ABORT MOUNT PROOF:$MOUNT_ERR"; row "mount$MOUNT_ERR"; exit 1; }
  say "mount proof ok (4/4 instrumented modules live)"
fi

# --- attestation (same rules the recorded table was gated on) ----------------
if ! attest "$L" "$CEILING" "${NUM_SPEC_TOKENS:-7}" "$([[ $KVDT == auto ]] && echo "" || echo "$KVDT")"; then
  say "ABORT ATTESTATION"; row attest-failed; exit 1
fi
ceil=$(grep -oE "max_cudagraph_capture_size.: [0-9]+" "$L" | tail -1 | grep -oE "[0-9]+$")
cap=$(grep -oE "Graph capturing finished in [0-9.]+ secs, took [0-9.]+ GiB" "$L" | sort -u | tr '\n' ';')
kvtok=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$L" | tail -1)
dtype=$(grep -oE "dtype=torch.[a-z0-9]+" "$L" | tail -1)
maxsize=$(grep -oE "cudagraph_capture_sizes.: \[[0-9, ]+\]" "$L" | tail -1 \
          | tr -d 'a-z_[] ' | tr ',' '\n' | grep -E '^[0-9]+$' | sort -n | tail -1)
say "healthy +$((j*3))s ceil=${ceil:-?} ladder_max=${maxsize:-?} capture=${cap:-none} | ${kvtok:-kv?} | ${dtype:-dtype?}"

# --- engagement assertion ----------------------------------------------------
CENSUS=$(grep -oE "\[w57-census\] (path|comp)=.*" "$L" | sort -u)
# A census-shaped needle can only be satisfied by the instrumented modules, so
# demanding one without PROBES= is a rig error, not an engine verdict. Calling
# that "the knob is inert" would be a false accusation -- and it is the mistake
# the W52 XPU_* knobs were reported under. Fail loudly as a rig problem instead.
if [[ -n "$NEEDLE" && -z "$PROBES" && -z "$CENSUS" ]] \
   && grep -qE '(^|[^a-z_])(comp|path)=' <<<"$NEEDLE"; then
  say "ABORT RIG: NEEDLE asks for a census line but PROBES= was not given -- the engine cannot be judged on instrumentation it never had"
  row "rig-needle-without-probes"; exit 2
fi
if [[ -n "$NEEDLE" ]] && ! { grep -qE "$NEEDLE" "$L" || grep -qE "$NEEDLE" <<<"$CENSUS"; }; then
  say "ABORT ENGAGEMENT: nothing matched /$NEEDLE/ -- the knob is INERT, not neutral"
  row "engage-missing:$NEEDLE"; exit 1
fi
# NEEDLE_ABSENT is only meaningful if the census was actually capable of
# producing that line; an un-instrumented engine matches nothing, so a passing
# absence check would be vacuous.
if [[ -n "$NEEDLE_ABSENT" ]] && { grep -qE "$NEEDLE_ABSENT" <<<"$CENSUS" \
    || { [[ -z "$PROBES" ]] && grep -qE '(^|[^a-z_])(comp|path)=' <<<"$NEEDLE_ABSENT"; }; }; then
  say "ABORT ENGAGEMENT: /$NEEDLE_ABSENT/ present, or its census was never mounted to prove it absent"
  row "engage-present:$NEEDLE_ABSENT"; exit 1
fi
[[ -n "$NEEDLE$NEEDLE_ABSENT" ]] && say "engagement assertion satisfied"

# --- gates: reuse the shipped harnesses, do not reimplement them -------------
cgate=PASS
PORT="$ARM_PORT" SERVED_NAME="$MNAME" REPS="${REPS:-3}" \
  timeout 900 bash "$HERE/correctness.sh" > "$O/${LABEL}.correctness" 2>&1 \
  || cgate=FAIL
cg_line=$(grep -oE '[0-9]+/[0-9]+' "$O/${LABEL}.correctness" | tail -1)
say "correctness ${cg_line:-?} ($cgate)"

agate=PASS
PORT="$ARM_PORT" SERVED_NAME="$MNAME" \
  timeout 900 bash "$HERE/acceptance.sh" > "$O/${LABEL}.acceptance" 2>&1 \
  || agate=FAIL
acc1=$(grep -oE 'C1-window[[:space:]]+[0-9]+ [0-9]+ [0-9.na/]+' "$O/${LABEL}.acceptance" | head -1 | sed 's/C1-window//;s/^ *//')
accn=$(grep -oE 'NAT-window[[:space:]]+[0-9]+ [0-9]+ [0-9.na/]+' "$O/${LABEL}.acceptance" | head -1 | sed 's/NAT-window//;s/^ *//')
say "acceptance: C1 '$acc1' | NAT '$accn' ($agate)"

GATE=PASS
[[ "$cgate" == PASS && "$agate" == PASS ]] || GATE=FAIL

# --- throughput at the MAX_NUM_SEQS cap, canonical budget --------------------
capn=${MAX_NUM_SEQS:-10}; midn=$((capn/2)); [[ $midn -ge 1 ]] || midn=1
conc=$(timeout 1800 python3 "$HERE/conc.py" --port "$ARM_PORT" --model "$MNAME" \
        --cells "$capn:$((capn*3)),$midn:$((midn*4)),1:6" --json 2>&1 | tail -1)
say "conc: $conc"
grep -q '"conc10"' <<<"$conc" || { say "INVALID: conc10 missing"; row conc-probe; exit 1; }
printf '%s\n' "$conc" > "$O/${LABEL}.conc.json"

# --- single-stream probes (same harness as the recorded rows) ----------------
c1=$(timeout 300 python3 "$HERE/probe.py" --url "http://127.0.0.1:$ARM_PORT/v1/completions" \
      --model "$MNAME" --prompt "1 2" --max-tokens 16 --iters 10 --warmup 3 \
      --label C1 2>&1 | tr -d '\n')
nat=$(timeout 300 python3 "$HERE/probe.py" --url "http://127.0.0.1:$ARM_PORT/v1/completions" \
      --model "$MNAME" --prompt "$(cat "$HERE/../../data/prompt-planning.txt")" \
      --max-tokens 96 --iters 6 --warmup 2 --label NAT 2>&1 | tr -d '\n')
printf '%s\n' "$c1"  > "$O/${LABEL}.c1raw.json"
printf '%s\n' "$nat" > "$O/${LABEL}.natraw.json"

pj(){ python3 -c "import json,sys
t=sys.stdin.read(); i=t.find('{')
try: print(json.dumps(json.loads(t[i:])))
except Exception: print('null')" <<<"$1"; }

# --- refuse to record an arm whose measurements did not actually happen --------
# The rules live in scripts/lib/degenerate.sh so the falsification harness tests
# the same code the live path runs. The reason one arm had to be rejected this way
# (gate=PASS, two conc cells at 0 tokens, probes unparseable) is recorded there.
degenerate_msg=$(assert_not_degenerate "$O/${LABEL}.conc.json" \
                                        "$O/${LABEL}.c1raw.json" \
                                        "$O/${LABEL}.natraw.json")
drc=$?
if [[ $drc -ne 0 ]]; then
  say "INVALID: $degenerate_msg -- not recording this arm as measured"
  row measurement-degenerate:${degenerate_msg#DEGENERATE: } "${PASS_ENV[*]:-}"
  exit 1
fi
printf '%s\n' "$CENSUS" > "$O/${LABEL}.census"
# Values travel by environment, not by positional argv. The first version passed
# 18 positional arguments and read one of them (the census path) at an index that
# belonged to kvtok, so open() was handed "GPU KV cache size: 384,956 tokens" and
# the writer died with FileNotFoundError -- appending nothing. Worse, `say "row
# recorded"` ran regardless of the exit status, so the log cheerfully reported a
# row that did not exist and the control arm looked recorded in a table that only
# held boot-fail rows. Env passing removes the index class of bug entirely, and
# the exit status is now checked.
A_LABEL="$LABEL" A_PORT="$ARM_PORT" A_ENV="$*" A_GATE="$GATE" \
A_CORRECTNESS="${cg_line:-0/0}" A_CGATE="$cgate" A_AGATE="$agate" \
A_C1ACC="$acc1" A_NATACC="$accn" A_C1RAW="$(pj "$c1")" A_NATRAW="$(pj "$nat")" \
A_CONC="$conc" A_CENSUS_FILE="$O/${LABEL}.census" A_KVTOK="${kvtok:-}" \
A_DTYPE="${dtype:-?}" A_CAPTURE="${cap:-none}" A_LADDER="${maxsize:-?}" \
python3 -c "
import json, os, sys
def j(v):
    v = v.strip()
    i = v.find('{')
    try: return json.loads(v[i:]) if i >= 0 else None
    except Exception: return None
def s(n): return os.environ.get(n, '')
try:
    census = open(s('A_CENSUS_FILE')).read().strip().replace(chr(10), ';')
except OSError:
    census = ''
print(json.dumps({
 'label': s('A_LABEL'), 'port': int(s('A_PORT') or 0), 'env': s('A_ENV'),
 'gate': s('A_GATE'), 'correctness': s('A_CORRECTNESS'),
 'cgate': s('A_CGATE'), 'agate': s('A_AGATE'),
 'c1acc': s('A_C1ACC'), 'natacc': s('A_NATACC'),
 'c1raw': j(s('A_C1RAW')), 'natraw': j(s('A_NATRAW')),
 'conc': j(s('A_CONC')) or s('A_CONC'),
 'kvtok': s('A_KVTOK'), 'dtype': s('A_DTYPE'),
 'capture': s('A_CAPTURE'), 'ladder_max': s('A_LADDER'), 'census': census}))" \
  >> "$ROWS"
row_rc=$?
if [[ $row_rc -ne 0 ]] || ! tail -1 "$ROWS" | python3 -c "import json,sys; json.loads(sys.stdin.read())" 2>/dev/null; then
  say "FATAL: row writer failed (rc=$row_rc) -- NOT reporting this arm as recorded"
  exit 1
fi
say "row recorded (gate=$GATE correctness=$cgate acceptance=$agate)"

if [[ "${W57_HOLD:-0}" == "1" ]]; then
  say "W57_HOLD=1: leaving $CTR up on port $ARM_PORT; clearing the teardown trap"
  trap - EXIT
  exit 0
fi
