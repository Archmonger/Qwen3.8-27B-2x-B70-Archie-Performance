#!/usr/bin/env bash
# Fetch the two models this repo serves, then apply the drafter config patch.
#
# Both are public (no token needed). The target is 29 GB and the drafter 2.6 GB,
# so pass an explicit --local-dir on a big pool and expect 33 GB total.
#
# Why the drafter needs a config edit at all: it publishes
# `architectures: ["DSparkDraftModel"]` plus `auto_map`. On the pinned image that
# resolves to no model-registry entry, so the boot dies at SpeculativeConfig
# validation before any weights load. profiles/fp8/model-patches/drafter-config.json
# is the corrected file -- it changes only `architectures` (to Qwen3DSparkModel)
# and drops `auto_map`, and setup.sh verifies byte-for-byte equivalence with it,
# because a hyperparameter drift there would silently change acceptance and
# invalidate every number in the README.
#
# Requires `hf` (huggingface_hub CLI): uvx hf ... , pip install huggingface_hub,
# or `brew install huggingface`.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${MODELS_DIR:-/mnt/cache/models}"
TARGET_DIR="${TARGET_DIR:-Qwen3.8-27B-FP8}"
DRAFTER_DIR="${DRAFTER_DIR:-qwen3.8-27b-dflash-drafter-fp8-b70}"
TARGET_REPO="${TARGET_REPO:-Qwen/Qwen3.8-27B-FP8}"
TARGET_REVISION="${TARGET_REVISION:-017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}"
DRAFTER_REPO="${DRAFTER_REPO:-rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70}"
DRAFTER_REVISION="${DRAFTER_REVISION:-94f8b9cc16571ae53ce0a34494265a8419962cf3}"
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir) MODELS_DIR="${2:-}"; shift 2 ;;
    --target-repo) TARGET_REPO="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    # Skip line 1: `grep '^#'` also matches the shebang, which prints as "!/usr/bin/env bash".
    -h|--help) sed -n '2,/^[^#]/p' "$0" | sed -e '/^[^#]/d' -e 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "download-models: unknown option $1" >&2; exit 2 ;;
  esac
done

HF=""
for c in "hf" "huggingface-cli"; do command -v "$c" >/dev/null 2>&1 && { HF="$c"; break; }; done
if [[ -z "$HF" ]]; then
  command -v uvx >/dev/null 2>&1 && HF="uvx hf" \
    || { echo "download-models: no 'hf' CLI found. pip install huggingface_hub (or use 'uvx hf')." >&2; exit 1; }
fi
echo "fetching with: $HF"

fetch() { # $1=repo $2=dir $3=revision $4=min_expected_bytes
  local repo="$1" dir="$2" rev="$3" min="$4"
  local path="$MODELS_DIR/$dir"
  if [[ -d "$path" ]] && [[ $(find "$path" -maxdepth 1 -name '*.safetensors' | wc -l) -gt 0 ]]; then
    echo "  present: $path ($(du -sh "$path" 2>/dev/null | cut -f1)) -- leaving alone"
    return 0
  fi
  echo "  -> $repo @ ${rev:0:12} into $path"
  [[ $DRY -eq 1 ]] && return 0
  # Pinned to a commit, not a branch: a moved main branch here would change the
  # measured configuration while every README number stayed the same.
  $HF download "$repo" --revision "$rev" --local-dir "$path" \
    || { echo "download-models: fetch failed for $repo" >&2; return 1; }
  local bytes; bytes=$(du -sb "$path" | cut -f1)
  (( bytes >= min )) || { echo "download-models: $path is $bytes bytes, expected >= $min (interrupted?)" >&2; return 1; }
}

echo "[models] target + drafter into $MODELS_DIR (33 GB total)"
fetch "$TARGET_REPO"  "$TARGET_DIR"  "$TARGET_REVISION"  25000000000 || exit 1
fetch "$DRAFTER_REPO" "$DRAFTER_DIR" "$DRAFTER_REVISION"  2500000000 || exit 1

if [[ $DRY -eq 1 ]]; then echo "[models] dry run; nothing written"; exit 0; fi

# Apply the drafter patch here too, so a hand-run fetch is not one silent boot
# failure away from working. setup.sh re-checks and can also apply it.
CFG="$MODELS_DIR/$DRAFTER_DIR/config.json"
if grep -q '"DSparkDraftModel"' "$CFG" || grep -q '"auto_map"' "$CFG"; then
  cp -n "$CFG" "$CFG.orig-upstream" 2>/dev/null || true
  cp "$HERE/../profiles/fp8/model-patches/drafter-config.json" "$CFG" \
    || { echo "download-models: cannot write $CFG" >&2; exit 1; }
  echo "[models] drafter config.json patched (architectures -> Qwen3DSparkModel, auto_map removed)"
else
  echo "[models] drafter config.json already patched"
fi

# The drafter's own safetensors sha256 is what makes "identical weights" a
# checkable claim rather than a hope.
want="b6c45ca365d983254c8f7a1dfb9c116f176b4ab0a8710180f810a08e2ad335cc"
SW="$MODELS_DIR/$DRAFTER_DIR/model.safetensors"
# Distinguish "absent" from "different": an unguarded sha256sum on a missing file
# prints nothing, and empty-vs-expected reads as a checksum mismatch, sending you
# to hunt for a substituted checkpoint when the download simply never landed.
if [[ ! -f "$SW" ]]; then
  echo "[models] FATAL $SW is missing -- the drafter download did not complete" >&2
  exit 1
fi
got=$(sha256sum "$SW" | cut -d' ' -f1)
if [[ "$got" == "$want" ]]; then
  echo "[models] ok  drafter weights match the measured run ($got)"
else
  echo "[models] WARN drafter weights differ from the measured run"
  echo "[models]      expected $want"
  echo "[models]      got      $got  -- every throughput number in the README is then unattributable"
fi
echo "[models] done. Next: sudo ../setup.sh   (from the repo root: sudo ./setup.sh)"
