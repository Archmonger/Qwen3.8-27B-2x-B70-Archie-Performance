#!/usr/bin/env bash
# Falsification harness for scripts/lib/kv-patch.sh, the gate setup.sh runs before
# it will start the host KV tier.
#
# Why this needs to exist at all: the tier is on in profiles/fp8, and the pairing
# "tier on + unpatched image" is what kills EngineCore on a long request. setup.sh
# is supposed to refuse it. A probe that can only ever print PATCHED makes that
# refusal decorative -- so this harness proves the probe CAN say UNPATCHED, against
# real images, not just against strings.
#
# Two layers, because they fail differently:
#   1. Classifier cases over synthetic text. Fast, needs no docker, and covers the
#      cases that actually bite: warnings on the same stdout stream, a diagnostic
#      that merely contains the word PATCHED, and an empty result.
#   2. Probe cases against real images, when docker and the pinned base are
#      present: the shipped image must read PATCHED, the unpatched base must read
#      UNPATCHED, and a mutated copy of the patched file must read UNPATCHED. The
#      mutation case is the one that matters -- it is what shows the probe is not
#      vacuously green.
#
# Usage:  scripts/falsify/kv-patch-cases.sh
#         SKIP_IMAGE_CASES=1 scripts/falsify/kv-patch-cases.sh   # classifier only
# Exit 0 only if every case behaves. An unavailable docker/base skips the image
# layer with a loud SKIP line, never a silent pass.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck source=../lib/kv-patch.sh
. "$REPO/scripts/lib/kv-patch.sh"

BASE_IMAGE="${BASE_IMAGE:-intel/llm-scaler-vllm@sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b}"
PATCHED_IMAGE="${PATCHED_IMAGE:-vllm-llmscaler:icr26.35}"
MUT_IMAGE="kvpatch-falsify-mutant:local"

fails=0
skips=0
ok()   { echo "ok   $*"; }
bad()  { echo "FAIL $*"; fails=$((fails + 1)); }
skip() { echo "SKIP $*"; skips=$((skips + 1)); }

# ---------------------------------------------------------------------------
# 1. Classifier cases. These are the ones that decide whether an unrunnable
#    probe can be mistaken for a verdict.
# ---------------------------------------------------------------------------
cls_case() { # $1=name $2=want $3=input
  local got
  got=$(kv_classify "$3")
  if [[ "$got" == "$2" ]]; then ok "classify $1 -> $got"
  else bad "classify $1: want $2 got $got (input=$(printf '%q' "$3"))"; fi
}

echo "=== kv-patch classifier ==="
cls_case "bare verdict            " PATCHED   $'PATCHED\n'
cls_case "verdict under a torch warning" PATCHED \
  $'/opt/venv/.../torch/xpu/__init__.py:74: UserWarning: XPU device count is zero!\n  return torch._C._xpu_getDeviceCount()\nPATCHED\n'
cls_case "bare UNPATCHED          " UNPATCHED $'UNPATCHED\n'
cls_case "UNPATCHED with noise    " UNPATCHED $'warning noise\nUNPATCHED\nmore noise\n'
# A diagnostic that merely contains the words must never attest the fix. This is
# the exact way a substring-matching probe goes permanently green.
cls_case "words inside prose      " UNKNOWN   $'note: image was UNPATCHED earlier but is PATCHED now\n'
cls_case "indented verdict        " UNKNOWN   $'   PATCHED\n'
cls_case "suffixed verdict        " UNKNOWN   $'PATCHED=yes\n'
cls_case "lookalike token         " UNKNOWN   $'NOT_PATCHED\n'
cls_case "empty output            " UNKNOWN   ''
cls_case "docker pull failure     " UNKNOWN   $'Unable to find image locally\ndocker: Error response from daemon: pull access denied\n'
cls_case "only a diagnostic       " UNKNOWN   $'kv-probe: missing /opt/venv/lib/python3.12/site-packages/vllm/x.py\n'
cls_case "both verdicts at once   " UNKNOWN   $'PATCHED\nUNPATCHED\n'

# ---------------------------------------------------------------------------
# 1b. Severity branch table. The part most likely to be quietly wrong: "fatal
#     when the tier is on, warn when it is off" is a table over two inputs, and
#     getting it backwards either blocks a working tier-off boot or lets the
#     crashing pairing through.
# ---------------------------------------------------------------------------
sev_case() { # $1=name $2=want $3=state $4=size
  local got
  got=$(kv_severity "$3" "$4")
  if [[ "$got" == "$2" ]]; then ok "severity $1 -> $got"
  else bad "severity $1: want $2 got $got"; fi
}

echo
echo "=== kv-patch severity table ==="
sev_case "patched, tier off           " ok    PATCHED   0
sev_case "patched, tier on            " ok    PATCHED   50
sev_case "patched, size unset         " ok    PATCHED   ''
sev_case "unpatched, tier off         " warn  UNPATCHED 0
sev_case "unpatched, size unset       " warn  UNPATCHED ''
sev_case "unpatched, tier on          " fatal UNPATCHED 50
sev_case "probe broke, tier off       " warn  UNKNOWN   0
sev_case "probe broke, tier on        " fatal UNKNOWN   50
# 0.0 is not the same as omitting the flag. serve.sh tests `!= "0"`, so it passes
# --kv-offloading-size 0.0, and config/vllm.py returns early only on None -- so an
# explicit 0.0 still builds OffloadingConnector. It must count as tier-on here.
sev_case "explicit 0.0 builds connector" fatal UNPATCHED 0.0
sev_case "explicit 0.0, patched       " ok    PATCHED   0.0

# ---------------------------------------------------------------------------
# 2. Probe cases against real images.
# ---------------------------------------------------------------------------
echo
echo "=== kv-patch probe (real images) ==="
if [[ "${SKIP_IMAGE_CASES:-0}" == "1" ]]; then
  skip "image cases requested off (SKIP_IMAGE_CASES=1)"
elif ! command -v docker >/dev/null 2>&1; then
  skip "no docker CLI on PATH"
elif ! docker info >/dev/null 2>&1; then
  skip "docker daemon unreachable (run as root?)"
else
  have() { [[ -n "$(docker images -q "$1" 2>/dev/null)" ]]; }
  # Build the mutant: same file, one bound removed. If the probe still says
  # PATCHED after this, the gate is decoration.
  build_mutant() {
    local d
    d="$(mktemp -d)"
    docker run --rm --entrypoint sh "$PATCHED_IMAGE" -c '
      V="$(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null | tail -n 1)"
      F="$V/'"$KV_PATCH_REL"'"
      sed "s/'"$KV_PATCH_MARK_BOUND"'/num_keyed_chunks = 0 # MUTATED/" "$F" > /f
      # fail loudly if the sed matched nothing: a mutant that equals the
      # original would "prove" the probe correct while testing nothing
      grep -q "MUTATED" /f || { echo "mutation did not apply" >&2; exit 9; }
      grep -q "'"$KV_PATCH_MARK_BOUND"'" /f && { echo "bound still present" >&2; exit 9; }
      cat /f
    ' > "$d/scheduler.py" || { rm -rf "$d"; return 1; }
    [[ -s "$d/scheduler.py" ]] || { rm -rf "$d"; return 1; }
    cat > "$d/Dockerfile" <<EOF
FROM $PATCHED_IMAGE
COPY scheduler.py /opt/venv/lib/python3.12/site-packages/vllm/$KV_PATCH_REL
EOF
    docker build -q -t "$MUT_IMAGE" "$d" >/dev/null || { rm -rf "$d"; return 1; }
    rm -rf "$d"
  }

  cleanup() { docker rmi -f "$MUT_IMAGE" >/dev/null 2>&1 || true; }
  # Overridden below when the gate section actually runs. Defaulting it to a no-op
  # means ONE trap definition can name both: leaving the profile mutated because an
  # EXIT trap was re-set for the mutant image alone is the kind of leak that turns a
  # red harness run into a broken next boot.
  gate_restore() { :; }
  trap 'cleanup; gate_restore' EXIT
  trap 'cleanup; gate_restore; exit 130' INT TERM

  probe_case() { # $1=name $2=want $3=image
    local out got
    out=$(kv_probe "$3"); got=$(kv_classify "$out")
    if [[ "$got" == "$2" ]]; then
      ok "probe $1 -> $got"
    else
      bad "probe $1: want $2 got $got"
      tail -3 <<<"$out" | sed 's/^/       /'
    fi
  }

  if have "$PATCHED_IMAGE"; then
    probe_case "shipped image           " PATCHED "$PATCHED_IMAGE"
    # The mutant is only a valid test if the thing it was built from is the real
    # patched file, so gate it on the case above having a real target.
    if [[ "$(kv_classify "$(kv_probe "$PATCHED_IMAGE")")" == "PATCHED" ]]; then
      if build_mutant; then
        probe_case "mutated bound (must fail)" UNPATCHED "$MUT_IMAGE"
      else
        bad "could not build the mutant image -- the probe's ability to say "
        bad "  UNPATCHED is therefore unproven on this box"
      fi
    fi
  else
    skip "image $PATCHED_IMAGE absent; cannot test the PATCHED path or the mutant"
  fi

  if have "$BASE_IMAGE" || have "${BASE_IMAGE%@*}"; then
    probe_case "unpatched pinned base     " UNPATCHED "$BASE_IMAGE"
  else
    skip "pinned base $BASE_IMAGE absent; cannot test the UNPATCHED path"
  fi

  probe_case "nonexistent image           " UNKNOWN "kvpatch-no-such-image:absent"

  # -------------------------------------------------------------------------
  # 3. End-to-end: does setup.sh actually REFUSE the crashing pairing?
  #    Cases 1-2 prove the probe reads correctly; this proves the probe is
  #    WIRED to a refusal. A correct probe that nothing acts on is decoration.
  #
  #    Runs the real setup.sh under --no-build --no-start --no-verify, so it
  #    cannot build an 18 GB image and cannot touch a running container. NOT
  #    --dry-run: the dry run exits at the compose render, above this gate.
  #    --no-build also pins the fatal branch, since DO_BUILD=0 skips the
  #    auto-rebuild and leaves the refusal.
  #
  #    The profile is mutated and restored, the same way setup-guards.sh does
  #    it. The CONTROL (case B) is what makes case A mean anything: if the
  #    control also fails, the box cannot run setup.sh at all (missing models,
  #    no docker group) and A's "refusal" proved nothing.
  # -------------------------------------------------------------------------
  echo
  echo "=== kv-patch gate (setup.sh end-to-end) ==="
  ENVF="$REPO/profiles/fp8/fp8.env"
  if [[ ! -f "$ENVF" ]]; then
    skip "no profile at $ENVF"
  elif [[ -n "${ARM_CHECK_ONLY:-}" ]]; then
    skip "not run under ARM_CHECK_ONLY"
  else
    GATE_BK="$(mktemp -d)/fp8.env"
    cp "$ENVF" "$GATE_BK"
    # Replaces the no-op defined above; the single EXIT/TERM trap now covers both
    # the mutant image and the mutated profile.
    gate_restore() { [[ -f "$GATE_BK" ]] && cp "$GATE_BK" "$ENVF"; }

    # $1=label $2=image $3=size  -> rc 0 means setup.sh accepted the pairing.
    # Captures output in GATE_OUT but stays silent: case A is SUPPOSED to exit
    # non-zero, so printing "exited 1" there reads like a fault on a passing
    # case. gate_case decides whether the failure is a verdict or a problem, and
    # only it prints.
    gate_run() { # $1=image $2=size
      local img="$1" size="$2" rc
      gate_restore
      sed -i "s|^IMAGE=.*|IMAGE=$img|" "$ENVF"
      sed -i "s|^KV_OFFLOAD_SIZE=.*|KV_OFFLOAD_SIZE=$size|" "$ENVF"
      # --no-drafter-patch matters: with DRY=0 and DO_START irrelevant, setup.sh
      # still reaches the drafter block and COPIES profiles/fp8/model-patches/
      # drafter-config.json over $MODELS_DIR/$DRAFTER_DIR/config.json (writing a
      # .orig-upstream next to it). Every other harness here runs --dry-run and
      # so never touched a model file; this one cannot, because the gate sits
      # below the dry-run exit. Refusing to mutate the deployment is the point.
      GATE_OUT="$(cd "$REPO" && timeout 300 ./setup.sh --profile fp8 \
                   --no-build --no-start --no-verify --no-drafter-patch 2>&1)"; rc=$?
      return $rc
    }

    # What setup.sh actually died on, printed only where a failure is a problem
    # rather than the expected verdict.
    gate_diag() {
      grep -m3 -E "FATAL|WARN: .*KV-offload" <<<"$GATE_OUT" | sed 's/^/         /'
    }

    gate_case() { # $1=name $2=want-accept(0|1) $3=image $4=size
      local name="$1" want="$2" img="$3" size="$4" rc=0
      if gate_run "$img" "$size"; then rc=0; else rc=1; fi
      if [[ "$want" == "1" ]]; then
        # A refusal only counts if it is THIS gate's refusal. Requiring the
        # fatal branch's own wording is what stops an unrelated preflight
        # failure (missing model dir, no docker group, a moved BDF) from being
        # tallied as the gate firing -- the exact vacuous-pass shape this
        # project has been bitten by. "tier is enabled" appears only in the
        # fatal branch, never in the warn branch.
        if [[ "$rc" == "1" ]] \
           && grep -q "KV-offload hotfix not confirmed" <<<"$GATE_OUT" \
           && grep -q "the host KV tier is enabled" <<<"$GATE_OUT"; then
          ok "gate $name -> refused by the KV gate"
        elif [[ "$rc" == "1" ]]; then
          bad "gate $name: setup.sh failed but NOT from the KV gate (its wording absent)"
          gate_diag
        else
          bad "gate $name: the crashing pairing was accepted"
        fi
      else
        if [[ "$rc" == "0" ]]; then
          ok "gate $name -> accepted"
        else
          bad "gate $name: expected acceptance but setup.sh exited $rc"
          gate_diag
        fi
      fi
    }

    # The gate needs a ref setup.sh itself can resolve with `docker images -q`,
    # because setup.sh dies with "image absent and --no-build was given" if it
    # cannot -- which would fail cases A and B for the wrong reason. The tag form
    # is what `docker images` lists; the digest form is not always, so the gate
    # uses UNPATCHED_IMAGE rather than the Dockerfile's digest-pinned BASE_IMAGE.
    UNPATCHED_IMAGE="${UNPATCHED_IMAGE:-intel/llm-scaler-vllm:0.26.0-b2}"
    if have "$UNPATCHED_IMAGE" && have "$PATCHED_IMAGE"; then
      # A: the pairing that kills EngineCore must be refused.
      gate_case "unpatched image + tier on   " 1 "$UNPATCHED_IMAGE" 50
      # B: control. Same image, tier off => must be accepted, or A proved nothing.
      gate_case "control: unpatched + off    " 0 "$UNPATCHED_IMAGE" 0
      # C: the shipping config must come up.
      gate_case "patched image + tier on     " 0 "$PATCHED_IMAGE" 50
      # D: a tier-off boot must never be blocked by this gate.
      gate_case "patched image + tier off    " 0 "$PATCHED_IMAGE" 0
    else
      skip "need both $UNPATCHED_IMAGE and $PATCHED_IMAGE to test the gate end-to-end"
    fi
    # Restore now rather than only at exit. The EXIT trap still calls
    # gate_restore, and it is idempotent, so the belt-and-braces costs nothing.
    gate_restore
  fi
fi

echo
if [[ $fails -eq 0 ]]; then
  echo "falsify/kv-patch: PASS -- the classifier cannot be fooled and the probe$(
    [[ $skips -gt 0 ]] && echo ' (image layer partially skipped)'
  ) distinguishes patched from unpatched from unprovable"
  [[ $skips -gt 0 ]] && echo "  ($skips skip(s): run on specter-nebula for the full layer)"
  exit 0
fi
echo "falsify/kv-patch: FAIL ($fails problem(s)) -- do not trust the KV-offload gate"
exit 1
