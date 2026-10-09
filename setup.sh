#!/usr/bin/env bash
# Stand up the Qwen3.8-27B FP8 + DSpark + XPU-graph serving container on 2x Intel
# Arc Pro B70 (TP=2).
#
#   git clone <this repo> && cd <this repo>
#   sudo ./setup.sh                      # preflight -> build -> generate -> start -> verify
#   sudo ./setup.sh --dry-run            # preflight + resolved plan, changes nothing
#   sudo ./setup.sh --profile int4       # placeholder: refuses, tells you why
#
# Root is needed to read /sys/class/drm and /dev/dri and to drive docker. Set
# MODELS_DIR / CACHE_DIR / RUNSTATE_DIR in the profile's .env (or pass
# --models-dir/--cache-dir/--run-dir) when your layout differs from this box's.
#
# Idempotent: re-running rebuilds nothing unless the image is missing, regenerates
# the runstate directory, and force-recreates the container.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE=fp8
DRY=0
DO_BUILD=1
FORCE_BUILD=0
DO_START=1
DO_VERIFY=1
APPLY_DRAFTER_PATCH=1

# Filled by load_profile.
ENV_FILE=""
IMAGE=""
MODELS_DIR="" CACHE_DIR="" RUNSTATE_DIR=""
B70_BDFS=""
MAX_NUM_SEQS="" NUM_SPEC_TOKENS="" CEILING=""
PORT="" SERVED_NAME="" TP_SIZE="" SPEC_METHOD=""
TARGET_DIR="" DRAFTER_DIR=""
RANK0_CARD="" RANK0_RENDER="" RANK1_CARD="" RANK1_RENDER=""

say()  { printf '[setup] %s\n' "$*"; }
warn() { printf '[setup] WARN: %s\n' "$*" >&2; }
die()  { printf '[setup] FATAL: %s\n' "$*" >&2; exit 1; }
# A preflight failure must name the check that failed, not just exit 1: several
# of these conditions produce a container that starts and answers /health while
# serving the wrong thing.
check() { # $1=description $2=0-means-ok
  if [[ "$2" -eq 0 ]]; then say "ok      $1"; else die "$1"; fi
}

usage() {
  # Print the leading comment block, stopping at the first non-comment line. A
  # hardcoded line range silently starts printing code as the header grows.
  sed -n '2,/^[^#]/p' "$0" | sed -e '/^[^#]/d' -e 's/^# \{0,1\}//'
  cat <<'EOF'

Options
  --profile NAME       fp8 (default) | int4 (placeholder, not implemented)
  --models-dir DIR     parent of the model directories        (env MODELS_DIR)
  --cache-dir DIR      parent of the vllm/triton cache dirs   (env CACHE_DIR)
  --run-dir DIR        where to generate runstate (by-path)   (env RUNSTATE_DIR)
  --b70-bdfs LIST      comma-separated PCI BDFs of the two B70s (env B70_BDFS)
  --no-build           skip the image build (fails if the image is absent)
  --rebuild            build even if the image already exists
  --no-start           stop after preflight/build/generate
  --no-verify          start, but skip scripts/verify.sh
  --no-drafter-patch   do not rewrite the drafter config.json if it needs it
  --dry-run            preflight + print the resolved plan; write and start nothing
  -h, --help           this text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)     PROFILE="${2:-}"; shift 2 ;;
    --models-dir)  _CLI_MODELS_DIR="${2:-}"; shift 2 ;;
    --cache-dir)   _CLI_CACHE_DIR="${2:-}"; shift 2 ;;
    --run-dir)     _CLI_RUNSTATE_DIR="${2:-}"; shift 2 ;;
    --b70-bdfs)    _CLI_B70_BDFS="${2:-}"; shift 2 ;;
    --no-build)    DO_BUILD=0; shift ;;
    --rebuild)     FORCE_BUILD=1; shift ;;
    --no-start)    DO_START=0; shift ;;
    --no-verify)   DO_VERIFY=0; shift ;;
    --no-drafter-patch) APPLY_DRAFTER_PATCH=0; shift ;;
    --dry-run)     DRY=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             usage >&2; die "unknown option: $1" ;;
  esac
done

# ---------------------------------------------------------------------------
# Profile dispatch. int4 is a placeholder on purpose: it has never been booted
# from this repo, so it must fail closed rather than start something unverified.
# ---------------------------------------------------------------------------
case "$PROFILE" in
  fp8)
    ENV_FILE="$REPO/profiles/fp8/fp8.env"
    COMPOSE_FILE="$REPO/profiles/fp8/docker-compose.yml"
    ;;
  int4)
    cat >&2 <<'EOF'
[setup] FATAL: the Qwen3.8-27B INT4 profile is a placeholder and is not
[setup]        implemented. profiles/int4/ holds no runnable config, so nothing
[setup]        unverified can be booted from this repo.
[setup]
[setup]        See profiles/int4/README.md for the shape the profile will take and
[setup]        the constraints it must respect, and docs/measurement-notes.md for
[setup]        why the INT4 figures from the investigation are not reusable here.
EOF
    exit 3
    ;;
  *) die "unknown profile: $PROFILE (fp8 | int4)" ;;
esac

[[ -f "$ENV_FILE" ]] || die "missing profile env file: $ENV_FILE"

# Read the profile env file for compose's own interpolation. compose ALSO reads it
# directly as env_file, so one file stays the single source of truth. The loader is
# shared with verify.sh/soak.sh/run-all.sh -- see scripts/lib/profile-env.sh for why
# it parses rather than sources.
. "$REPO/scripts/lib/profile-env.sh"
load_env_file "$ENV_FILE"

# CLI wins over the env file.
[[ -n "${_CLI_MODELS_DIR:-}"    ]] && MODELS_DIR="$_CLI_MODELS_DIR"
[[ -n "${_CLI_CACHE_DIR:-}"     ]] && CACHE_DIR="$_CLI_CACHE_DIR"
[[ -n "${_CLI_RUNSTATE_DIR:-}"  ]] && RUNSTATE_DIR="$_CLI_RUNSTATE_DIR"
[[ -n "${_CLI_B70_BDFS:-}"      ]] && B70_BDFS="$_CLI_B70_BDFS"
MODELS_DIR="${MODELS_DIR:-/mnt/cache/models}"
CACHE_DIR="${CACHE_DIR:-/mnt/cache/.cache}"
RUNSTATE_DIR="${RUNSTATE_DIR:-runstate}"
TARGET_DIR="${TARGET_DIR:-Qwen3.8-27B-FP8}"
DRAFTER_DIR="${DRAFTER_DIR:-qwen3.8-27b-dflash-drafter-fp8-b70}"
B70_BDFS="${B70_BDFS:-0000:03:00.0,0000:83:00.0}"
# The profile keeps RUNSTATE_DIR relative because compose resolves a relative bind
# source against the compose file's own directory. Anchor it to the same place here
# so setup.sh and a manual `docker compose up -d` generate and mount the identical
# directory -- if they disagreed, the container would mount a by-path dir nobody
# regenerated after the last GPU reorder.
case "$RUNSTATE_DIR" in
  /*) ;;
  *) RUNSTATE_DIR="$(cd "$(dirname "$COMPOSE_FILE")" && pwd)/$RUNSTATE_DIR" ;;
esac
# Collapse the ./ the profile must keep for compose (see fp8.env), so paths printed
# and created are clean.
RUNSTATE_DIR="${RUNSTATE_DIR//\/.\//\/}"
export MODELS_DIR CACHE_DIR RUNSTATE_DIR TARGET_DIR DRAFTER_DIR IMAGE

# ---------------------------------------------------------------------------
# Preflight: host, driver, devices, models, config self-consistency.
# ---------------------------------------------------------------------------
say "--- preflight (profile: $PROFILE) ---"

command -v docker >/dev/null 2>&1; check "docker CLI on PATH" $?
docker info >/dev/null 2>&1;       check "docker daemon reachable (are you root?)" $?
docker compose version >/dev/null 2>&1
check "docker compose plugin present" $?
# Checked explicitly because two of the gates below are python3 heredocs whose
# `|| die` message names the gate, not the missing interpreter. Without python3
# the drafter diff reports "drafter config check failed" and the ceiling rule
# reports "graph-ceiling rule violated" -- both of which read as a bad profile
# rather than as an absent dependency, and one of them is the check that would
# otherwise have caught the 38% eager fallback.
command -v python3 >/dev/null 2>&1
check "python3 on PATH (drives the drafter-config and graph-ceiling gates)" $?

# The two B70 BDFs. Accept 0000:03:00.0 or 03:00.0; canonicalise to domain form.
IFS=',' read -r -a BDFS <<<"$B70_BDFS"
[[ ${#BDFS[@]} -eq 2 ]] || die "B70_BDFS must list exactly 2 PCI BDFs (got: $B70_BDFS)"
say "ok      B70_BDFS lists 2 devices: ${BDFS[*]}"
for i in 0 1; do
  b="${BDFS[$i]}"
  [[ "$b" =~ ^([0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$ ]] \
    || die "malformed PCI BDF: $b"
  [[ "$b" == *:*:* ]] || b="0000:$b"
  BDFS[$i]="$b"
  # Do not `cat` this path: driver is a symlink to a *directory*, so cat exits
  # non-zero and a `... && basename ...` chain silently yields an empty string,
  # which then reports a healthy xe card as "driver is 'none'".
  drv="$(basename "$(readlink -f "/sys/bus/pci/devices/$b/driver" 2>/dev/null)" 2>/dev/null)"
  [[ "$drv" == "xe" ]] || die "$b driver is '${drv:-none}', expected xe (the B70s are the only xe devices; check --b70-bdfs)"
  say "ok      $b bound to xe"
  # Resolve cardN/renderDN from the by-path names, not by scanning /dev/dri:
  # render node numbering is not stable across boots or across GPUs present.
  for kind in card render; do
    link="/dev/dri/by-path/pci-$b-$kind"
    [[ -e "$link" ]] || die "missing $link (is the xe driver loaded?)"
    node="$(basename "$(readlink -f "$link")")"
    var="RANK${i}_$( [[ $kind == card ]] && echo CARD || echo RENDER )"
    printf -v "$var" '%s' "$node"
    say "ok      $b $kind -> $node"
  done
done
export RANK0_CARD RANK0_RENDER RANK1_CARD RANK1_RENDER
[[ "$RANK0_CARD" != "$RANK1_CARD" ]] || die "both BDFs resolved to the same card ($RANK0_CARD)"

# Never hand oneCCL a directory containing foreign DRM entries -- it aborts in
# fill_device_fds before CCL tuning ever matters. That is why this compose file
# mounts a generated directory instead of the host's /dev/dri/by-path.
foreign="$(ls /dev/dri/by-path/ 2>/dev/null | grep -vF -e "pci-${BDFS[0]}-" -e "pci-${BDFS[1]}-" | head -3 | tr '\n' ' ')"
if [[ -n "$foreign" ]]; then
  say "note    host /dev/dri/by-path also holds: $foreign -> excluded from the container"
fi

# Model presence is checked by file, not by directory: a half-finished
# `hf download` leaves a directory that looks present and fails at load.
need=(config.json)
[[ -d "$MODELS_DIR/$TARGET_DIR" ]] || die "target model dir missing: $MODELS_DIR/$TARGET_DIR"
for f in "${need[@]}"; do [[ -f "$MODELS_DIR/$TARGET_DIR/$f" ]] || die "target model missing $f"; done
nshards=$(find "$MODELS_DIR/$TARGET_DIR" -maxdepth 1 -name '*.safetensors' | wc -l)
[[ $nshards -ge 60 ]] || die "target model has $nshards safetensors files, expected 64+ (interrupted download?)"
say "ok      target model: $TARGET_DIR ($nshards safetensors)"
[[ -d "$MODELS_DIR/$DRAFTER_DIR" ]] || die "drafter dir missing: $MODELS_DIR/$DRAFTER_DIR"
[[ -f "$MODELS_DIR/$DRAFTER_DIR/model.safetensors" ]] || die "drafter missing model.safetensors"
say "ok      drafter: $DRAFTER_DIR"

# The drafter ships architectures DSparkDraftModel + auto_map. On 0.26 that
# resolves to no registry entry; this profile needs Qwen3DSparkModel with
# auto_map removed. Everything else in the file must stay byte-identical.
DRAFT_CFG="$MODELS_DIR/$DRAFTER_DIR/config.json"
if grep -q '"DSparkDraftModel"' "$DRAFT_CFG" || grep -q '"auto_map"' "$DRAFT_CFG"; then
  if [[ $APPLY_DRAFTER_PATCH -eq 1 && $DRY -eq 0 ]]; then
    say "drafter config.json is the upstream variant -> applying profile patch"
    cp -n "$DRAFT_CFG" "$DRAFT_CFG.orig-upstream" 2>/dev/null || true
    cp "$REPO/profiles/fp8/model-patches/drafter-config.json" "$DRAFT_CFG" \
      || die "could not write $DRAFT_CFG (read-only mount? pass --no-drafter-patch and edit by hand)"
  else
    warn "drafter config.json still upstream. With this image that fails at startup with"
    warn "  \"Model architectures ['DSparkDraftModel'] are not supported for now\"."
    warn "  Fix: cp profiles/fp8/model-patches/drafter-config.json $DRAFT_CFG"
    [[ $DRY -eq 0 ]] && die "refusing to start a container that cannot load the drafter"
  fi
else
  say "ok      drafter config.json already patched"
fi
# Guard the patch itself: hyperparameters drifting from upstream would silently
# change acceptance and make any measured number meaningless.
python3 - "$DRAFT_CFG" "$REPO/profiles/fp8/model-patches/drafter-config.upstream.json" <<'PY' || die "drafter config check failed"
import json, sys
live = json.load(open(sys.argv[1])); up = json.load(open(sys.argv[2]))
want = dict(up, architectures=["Qwen3DSparkModel"])
want.pop("auto_map", None)
if live != want:
    diff = {k: (live.get(k), want.get(k)) for k in set(live) | set(want) if live.get(k) != want.get(k)}
    print("[setup] FATAL: drafter config.json differs from the expected patched config in:", diff, file=sys.stderr)
    sys.exit(1)
print("[setup] ok      drafter config.json matches profiles/fp8/model-patches/ exactly")
PY

# The ceiling rule is the single highest-value check in this script. A ceiling
# below MAX_NUM_SEQS x (1+k) silently drops the saturated batch to eager: same
# acceptance counters, 38% less throughput, and nothing in the log says so.
python3 - "$MAX_NUM_SEQS" "$NUM_SPEC_TOKENS" "$CEILING" "$EXTRA_VLLM_ARGS" <<'PY' || die "graph-ceiling rule violated"
import json, re, sys
max_seqs, k, ceil, extra = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
want = max_seqs * (1 + k)
m = re.search(r'max_cudagraph_capture_size"?:(\d+)', extra)
in_args = int(m.group(1)) if m else None
if ceil != want:
    print(f"[setup] FATAL: CEILING={ceil} but MAX_NUM_SEQS({max_seqs}) x (1+k={k}) = {want}", file=sys.stderr)
    sys.exit(1)
if in_args != want:
    print(f"[setup] FATAL: EXTRA_VLLM_ARGS max_cudagraph_capture_size={in_args} but must be {want}", file=sys.stderr)
    sys.exit(1)
if "FULL_DECODE_ONLY" not in extra:
    print("[setup] FATAL: EXTRA_VLLM_ARGS must set cudagraph_mode FULL_DECODE_ONLY", file=sys.stderr)
    sys.exit(1)
print(f"[setup] ok      graph ceiling {want} = MAX_NUM_SEQS x (1+k), FULL_DECODE_ONLY")
PY

if [[ "${SPEC_TOKENS_SCHEDULE:-}" != "" ]]; then
  die "SPEC_TOKENS_SCHEDULE is set. With graphs on it garbles output (6-13/30 correct) while acceptance counters stay healthy. Read docs/cudagraph-stride-defect.md first."
fi
[[ "$SPEC_METHOD" == "dspark" ]] || warn "SPEC_METHOD=$SPEC_METHOD: dflash does not boot with this drafter on 0.26 (see docs/dspark-drafter.md)"

say "ok      models dir $MODELS_DIR, caches $CACHE_DIR, runstate $RUNSTATE_DIR"

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
say "--- plan ---"
say "  image          $IMAGE"
say "  compose        $COMPOSE_FILE"
say "  container      ${CONTAINER_NAME:-vllm-arch-$PROFILE} on port $PORT as '$SERVED_NAME' (TP=$TP_SIZE)"
say "  target         $MODELS_DIR/$TARGET_DIR"
say "  drafter        $MODELS_DIR/$DRAFTER_DIR"
say "  spec           method=$SPEC_METHOD k=$NUM_SPEC_TOKENS, ceiling $CEILING, no schedule"
say "  devices        $RANK0_RENDER + $RANK1_RENDER (cards $RANK0_CARD/$RANK1_CARD)"
if [[ $DRY -eq 1 ]]; then
  say "  DRY RUN: nothing below this line will happen"
fi

# Compose is always invoked with BOTH env files: the profile, and the generated
# devices.env that carries the resolved card/render node names. Using the same
# invocation here as the README documents for the manual path means a broken
# devices.env generation fails at this check instead of at container creation.
compose_files=(--env-file "$ENV_FILE")

write_devices_env() { # $1=path
  {
    echo "# Generated by setup.sh on $(date -u +%FT%TZ). Do not edit or commit;"
    echo "# regenerated from B70_BDFS=$B70_BDFS on every run. Compose reads this to"
    echo "# interpolate the devices: list."
    echo "RANK0_CARD=$RANK0_CARD"
    echo "RANK0_RENDER=$RANK0_RENDER"
    echo "RANK1_CARD=$RANK1_CARD"
    echo "RANK1_RENDER=$RANK1_RENDER"
  } > "$1"
}

if [[ $DRY -eq 1 ]]; then
  # Render the same way, from a temp file, so the dry run still writes nothing.
  DRY_DEV="$(mktemp)"
  trap 'rm -f "$DRY_DEV"' EXIT
  write_devices_env "$DRY_DEV"
  compose_files+=(--env-file "$DRY_DEV")
  rendered=$(docker compose "${compose_files[@]}" -f "$COMPOSE_FILE" config 2>&1) \
    && say "ok      compose file renders" || die "compose file does not render: $COMPOSE_FILE"
  # A blanked node name renders as a devices: entry for "", which fails at creation.
  # Catch it here, where the cause is still obvious.
  if grep -qE 'source: /dev/dri/$|source: ""' <<<"$rendered"; then
    die "compose rendered an empty device node -- check RANK*_CARD / RANK*_RENDER resolution"
  fi
  # Match the mount TARGET, not the source path: RUNSTATE_DIR is user-settable, so
  # grepping for a literal "runstate/" would fail a correct custom layout.
  grep -q "target: /dev/dri/by-path" <<<"$rendered" \
    || die "compose did not mount the generated by-path directory at /dev/dri/by-path"
  exit 0
fi

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
if [[ -z "$(docker images -q "$IMAGE" 2>/dev/null)" ]] || [[ $FORCE_BUILD -eq 1 ]]; then
  if [[ $DO_BUILD -eq 0 ]]; then
    die "image $IMAGE is absent and --no-build was given. Build it: docker build -t $IMAGE image/"
  fi
  say "--- building $IMAGE (18 GB base, a few minutes; pulls ~24 GB on first run) ---"
  docker build -t "$IMAGE" "$REPO/image" || die "image build failed"
else
  say "ok      image $IMAGE present (use --rebuild to force)"
fi
# Fail here with a clear message rather than inside the container with a DEVICE_LOST
# during autotune: the overlay is what makes the userspace driver match the host
# kernel's xe driver.
#
# --entrypoint is mandatory, not stylistic. The image's own ENTRYPOINT is
#   bash -c "source ... && exec vllm serve \$@"
# so `docker run --rm IMAGE python -c ...` hands `python` to vllm as a model path and
# dies with "Failed to infer device type" -- which reads as a broken driver overlay
# and is actually the check never having run. Override it, and separate "the probe did
# not run" from "the driver is genuinely broken", or a false negative here sends
# someone chasing the Dockerfile.
driver_check=$(docker run --rm --entrypoint python "$IMAGE" -c \
  "import ctypes; ctypes.CDLL('libze_intel_gpu.so.1'); print('driver loads OK')" 2>&1)
if grep -q "driver loads OK" <<<"$driver_check"; then
  say "ok      in-image GPU userspace driver loads"
else
  say "FATAL: the in-image level-zero driver probe failed. Output was:"
  tail -5 <<<"$driver_check" | sed 's/^/[setup]          /' >&2
  die "image $IMAGE cannot load libze_intel_gpu.so.1 -- the Dockerfile overlay did not apply, or the base/driver tags moved"
fi

# ---------------------------------------------------------------------------
# Runstate: the filtered by-path directory + cache dirs
# ---------------------------------------------------------------------------
say "--- generating runstate in $RUNSTATE_DIR ---"
BP="$RUNSTATE_DIR/dri-by-path"
# Rebuilt from scratch every run: a stale symlink into a device that moved would
# mount a foreign GPU and oneCCL would abort on it.
rm -rf "$BP"
mkdir -p "$BP" "$CACHE_DIR/vllm" "$CACHE_DIR/torchinductor" "$CACHE_DIR/triton" \
  || die "cannot create $RUNSTATE_DIR or $CACHE_DIR"
for i in 0 1; do
  b="${BDFS[$i]}"
  card_var="RANK${i}_CARD"; render_var="RANK${i}_RENDER"
  ln -sf "../${!card_var}"   "$BP/pci-$b-card"
  ln -sf "../${!render_var}" "$BP/pci-$b-render"
done
check "generated $(ls "$BP" | wc -l) by-path entries (expect 4)" \
      "$([[ $(ls "$BP" | wc -l) -eq 4 ]] && echo 0 || echo 1)"

# Validate the links by TARGET NAME, not by resolvability here. The links read
# `../card2`, which is meant to resolve INSIDE the container, where this directory is
# mounted at /dev/dri/by-path and `../card2` is the /dev/dri/card2 the `devices:` list
# passes in. On the host they are always dangling, so `[[ -e ]]` here was a false
# negative that aborted a good setup. What actually matters is that each target names
# a node that exists on the host, because that is what has to be passed through.
for l in "$BP"/*; do
  tgt="$(readlink "$l")"
  case "$tgt" in
    ../*) node="${tgt#../}" ;;
    *)    die "by-path entry $l has an unexpected target '$tgt' (expected ../<node>)" ;;
  esac
  [[ -e "/dev/dri/$node" ]] \
    || die "by-path entry $(basename "$l") -> ../$node, but /dev/dri/$node does not exist on the host"
done
say "ok      all $(ls "$BP" | wc -l | xargs) by-path targets exist as /dev/dri nodes (they resolve inside the container)"

# Write the resolved node names where compose can pick them up for interpolation, so
# the documented manual `docker compose up -d` path does not need to know them. They
# are deliberately NOT in fp8.env: card/render numbering is not stable across boots,
# and a stale value there would mount the wrong GPUs. Regenerated every run, from the
# same helper the --dry-run render check uses, so the two cannot disagree.
DEVICES_ENV="$RUNSTATE_DIR/devices.env"
write_devices_env "$DEVICES_ENV" || die "cannot write $DEVICES_ENV"
say "ok      wrote $DEVICES_ENV"
compose_files+=(--env-file "$DEVICES_ENV")

# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------
CTR="${CONTAINER_NAME:-vllm-arch-$PROFILE}"
if [[ $DO_START -eq 1 ]]; then
  # Two engines holding both GPUs produce a misleading
  # "Free memory on device xpu:1 ... less than desired" / OUT_OF_RESOURCES, which
  # reads like a VRAM problem and is not one.
  if docker ps -a --format '{{.Names}}' | grep -qx "$CTR"; then
    say "removing existing container $CTR"
    docker rm -f "$CTR" >/dev/null 2>&1 || die "could not remove $CTR"
  fi
  say "--- starting $CTR ---"
  docker compose "${compose_files[@]}" -f "$COMPOSE_FILE" up -d --force-recreate \
    || die "compose up failed"
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
if [[ $DO_VERIFY -eq 1 ]]; then
  say "--- verifying (attestation + correctness + acceptance + serving surface) ---"
  bash "$REPO/scripts/verify.sh" --profile "$PROFILE" || die "verification failed -- see the output above"
fi

say "done."
cat <<EOF

  endpoint   http://127.0.0.1:$PORT/v1  (model name: $SERVED_NAME)
  logs       docker logs -f $CTR
  stop       docker compose ${compose_files[*]} -f $COMPOSE_FILE down
  re-verify  ./scripts/verify.sh --profile $PROFILE
  benchmark  ./scripts/bench/run-all.sh --profile $PROFILE   (engine must be idle)
EOF
