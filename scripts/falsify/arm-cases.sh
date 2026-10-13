#!/usr/bin/env bash
# Falsification harness for scripts/bench/arm.sh.
#
# arm.sh is the thing that decides whether a measured difference is real, so a
# silent failure in it does not announce itself -- it produces a confident table.
# Eight defects were found in it by actually running it, or by reviewing what it
# recorded, and every one is asserted here, because each would have passed while
# broken:
#
#   1. the profile env is sourced AFTER argument parsing, so profiles/fp8/fp8.env
#      (PORT=8103, IMAGE=, CEILING=, GPU_MEM_UTIL=, ...) overwrote the caller's
#      overrides: every arm booted on the production port, and an A/B silently
#      became two identical arms. Asserted by asking for an override and reading
#      back what the resolved rig says.
#   2. the boot-watcher's fatal pattern contained the bare phrase
#      "Free memory on device", which gpu_worker also prints on a healthy INFO
#      line at every startup -- so healthy boots were aborted at ~250s. Asserted
#      both ways against the two real log lines.
#   3. the row writer read an 18-argument argv at the wrong index and died with
#      FileNotFoundError while the script still printed "row recorded". Asserted
#      by running the real writer block on the values that killed it, including
#      the two ways it must degrade rather than vanish.
#   4. rig errors (bad device node, missing filtered by-path dir) surfaced only
#      as BOOT FAILED, after the caller had already stopped production. Asserted
#      by ARM_CHECK_ONLY exiting 2 on each.
#   5. probe modules were validated after the ARM_CHECK_ONLY exit, so a tampered
#      or partial instrument directory passed the very dry run callers use to stay
#      safe. Asserted on three bad directories plus the two good ones.
#   6. MODELS_DIR/TARGET_DIR/DRAFTER_DIR defaulted to the author's machine, so a
#      profile that omitted them silently mounted some other checkpoint and
#      recorded its throughput as the profile's. Asserted with a VALID profile
#      that omits exactly those keys -- a nonexistent profile is rejected earlier
#      for an unrelated reason, which is how a first draft of this case passed
#      while testing nothing.
#   7. a misspelled knob (DTIPE=float16) fell through to the pass-through env list,
#      so the container booted at the profile default while the row was labelled
#      and env-stamped as the fp16 arm. Asserted by refusing it, and by accepting
#      every legitimate form so the refusals are not blanket.
#   8. an arm recorded gate=PASS with two conc cells at 0 tokens and unparseable
#      probes, because the only structural test asked whether the string "conc10"
#      appeared. Asserted against the real artifacts, with a mutation per rule
#      showing each one is load-bearing. A probe that survived 1 of 10 iterations
#      also passed on a healthy-looking median, so the iteration count is asserted
#      too -- again with a mutation.
#
# Nothing here touches docker run, a GPU, or a serving endpoint: ARM_CHECK_ONLY
# short-circuits before any container is created, and the rest is pattern and
# string work. The harness runs anywhere python3 does.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$HERE/../.."
ARM="$HERE/../bench/arm.sh"

fails=0
ok(){ echo "  ok   $*"; }
bad(){ echo "  FAIL $*"; fails=$((fails+1)); }
need_python(){ command -v python3 >/dev/null 2>&1 || { echo "  SKIP python3 not found"; return 1; }; return 0; }

# --- rig fixture -------------------------------------------------------------
# arm.sh resolves the profile relative to itself, so the fixture is supplied
# through the two overrides arm.sh documents -- DEVICES_ENV and an absolute
# RUNSTATE_DIR (both are in its caller-override list, which is what makes them
# survive the profile sourcing being tested in case 1).
# Cleanup is a FUNCTION, and runs on EXIT *and* INT/TERM.
#
# Two separate reasons it is shaped this way:
#   * A Ctrl-C during a long run must not leave the arm.sh mutant in
#     scripts/bench/. It has to live beside arm.sh (arm.sh resolves
#     lib/profile-env.sh from BASH_SOURCE), so it is not inside $T and an
#     EXIT-only trap only helps if bash reaches an exit normally.
#   * Interpolating the mutant path into a trap STRING does not survive quoting:
#     `trap 'rm -rf "x.'"$$"'.sh"' EXIT` opens a quote that the expansion never
#     closes, and bash reports `unexpected EOF while looking for matching "` at
#     the first signal. That was the shape here, and it was silently doing no
#     cleanup at all -- verified on the host by touching the file, signalling
#     the script, and seeing the file still there. A function needs no quoting.
MUT="$HERE/../bench/arm-mut.$$.sh"
T=$(mktemp -d)
w57_cleanup(){ rm -rf "$T" "$REPO/profiles/w57-nofp" "$MUT"; }
trap w57_cleanup EXIT
trap 'trap - EXIT; w57_cleanup; exit 130' INT TERM
[[ -f "$REPO/profiles/fp8/fp8.env" ]] || { echo "  FAIL: no profiles/fp8/fp8.env"; exit 1; }
mkdir -p "$T/runstate/dri-by-path"

# The device check only asserts the four paths exist, so any real nodes will do.
mapfile -t EXIST < <(ls /dev/dri 2>/dev/null | grep -E '^(card|renderD)[0-9]+$' | head -4)
if [[ ${#EXIST[@]} -ge 4 ]]; then
  printf 'RANK0_CARD=%s\nRANK0_RENDER=%s\nRANK1_CARD=%s\nRANK1_RENDER=%s\n' \
    "${EXIST[0]}" "${EXIST[1]}" "${EXIST[2]}" "${EXIST[3]}" > "$T/devices.env"
else
  echo "  SKIP device-dependent cases: fewer than 4 nodes under /dev/dri"
  SKIP_DEV=1
fi
rig(){ # run arm.sh in check mode with the fixture rig
  ARM_CHECK_ONLY=1 DEVICES_ENV="$T/devices.env" RUNSTATE_DIR="$T/runstate" "$@"
}

# --- 1. caller overrides must beat the profile -------------------------------
# The whole point of arm.sh is that the knob changed. If the profile wins, the
# two arms are identical and the A/B measures nothing while reporting a result.
if [[ "${SKIP_DEV:-0}" != 1 ]]; then
  out=$(rig bash "$ARM" CHECK 8199 2>&1)
  grep -q "port=8199" <<<"$out" \
    && ok "caller's port survives the profile (PORT=8103 did not win)" \
    || bad "port was clobbered by the profile: $out"

  out=$(rig env IMAGE=custom/img:1 CEILING=56 bash "$ARM" CHECK 8199 2>&1)
  grep -q "image=custom/img:1" <<<"$out" \
    && ok "IMAGE override survives the profile" \
    || bad "IMAGE was clobbered by the profile: $out"
  grep -q "ceiling=56" <<<"$out" \
    && ok "CEILING override survives the profile" \
    || bad "CEILING was clobbered (attestation would then check the wrong ceiling): $out"

  out=$(rig env DTYPE=bfloat16 bash "$ARM" CHECK 8199 2>&1)
  grep -q "dtype=bfloat16" <<<"$out" \
    && ok "DTYPE set in the environment is not cleared by the arg parser" \
    || bad "environment DTYPE was dropped: $out"

  out=$(rig env DTYPE=bfloat16 bash "$ARM" CHECK 8199 DTYPE=float16 2>&1)
  grep -q "dtype=float16" <<<"$out" \
    && ok "positional KEY=VAL beats the environment" \
    || bad "positional override lost to the environment: $out"

  # --- 1b. profile-pinned dtype knobs ---------------------------------------
  # fp8.env now SHIPS DTYPE and MAMBA_SSM_CACHE_DTYPE pins, and arm.sh seeds
  # $MAMBA_SSM from the profile pin. Without the seed, serve.sh still emits
  # --mamba-ssm-cache-dtype from the container env while $MAMBA_SSM stays empty,
  # so assert_ssm_applied skips itself on every profile-default arm -- an
  # unasserted dtype claim on the baseline itself. This is the hole the
  # MAMBA_SSM_CACHE_DTYPE refusal closed for arm arguments, reopened by the
  # profile file; the seed closes it again and this case keeps it closed.
  out=$(rig env MAMBA_SSM_CACHE_DTYPE=bfloat16 bash "$ARM" CHECK 8199 2>&1)
  grep -q "ssm=bfloat16" <<<"$out" \
    && ok "profile-pinned MAMBA_SSM_CACHE_DTYPE is seeded into the asserted ssm value" \
    || bad "profile ssm pin not seeded (assertion would skip on every default arm): $out"

  # The mutant must sit in arm.sh's OWN directory: arm.sh resolves
  # lib/profile-env.sh relative to BASH_SOURCE, so a copy under $T cannot even
  # load its profile parser (which is exactly the class of harness bug that
  # reports "still rejected" for a rule that is load-bearing). It is invoked as
  # `bash "$MUT"`, so the 644 a redirection creates does not matter, and
  # w57_cleanup removes it on every exit path. $MUT is defined up with the trap.
  if ! sed 's/^DTYPE=${DTYPE:-}; MAMBA_SSM=${MAMBA_SSM_CACHE_DTYPE:-};/DTYPE=${DTYPE:-}; MAMBA_SSM="";/' \
       "$ARM" > "$MUT" 2>"$T/mutseed.sed.err"; then
    bad "MUTATION profile ssm seed reverted: sed failed -- $(head -1 "$T/mutseed.sed.err")"
  elif cmp -s "$MUT" "$ARM"; then
    bad "MUTATION profile ssm seed reverted: sed matched nothing -- mutant IS the original"
  else
    out=$(rig env MAMBA_SSM_CACHE_DTYPE=bfloat16 bash "$MUT" CHECK 8199 2>&1)
    rm -f "$MUT"   # before the asserts: a FAIL here must not leave a mutant in the tree
    grep -q "ssm=profile" <<<"$out" \
      && ok "MUTATION profile ssm seed reverted: pin vanishes from the assertion -- rule is load-bearing" \
      || bad "MUTATION profile ssm seed reverted: still asserted ($out) -- the case above does not depend on the seed"
  fi

  # An explicit auto restores the omitted-flag control arm and must be VISIBLE as
  # auto, not collapse to the empty skip that prints "profile".
  out=$(rig bash "$ARM" CHECK 8199 MAMBA_SSM=auto 2>&1)
  grep -q "ssm=auto" <<<"$out" \
    && ok "MAMBA_SSM=auto is a visible control spelling, not an empty skip" \
    || bad "MAMBA_SSM=auto not honoured: $out"
fi

# --- 2. the fatal pattern must fail both ways --------------------------------
# Two-sided on purpose: a pattern that matches everything aborts healthy boots
# (the bug that cost a production downtime window), and one that matches nothing
# lets a wedged engine hang until the boot timeout.
FATAL=$(sed -n "s/^FATAL='\(.*\)'$/\1/p" "$ARM")
[[ -n "$FATAL" ]] || { bad "could not extract FATAL from arm.sh"; }
HEALTHY='(Worker_TP0 pid=1352) INFO 10-11 07:05:05 [gpu_worker.py:857] Free memory on device (29.18/30.3 GiB) on startup. Desired GPU memory utilization is (0.92, 27.87)'
DYING='(Worker pid=939) ERROR 10-11 05:31:21 [multiproc_executor.py:901] ValueError: Free memory on device xpu:0 (0.82/31.89 GiB) on startup is less than desired GPU memory utilization'
if [[ -n "$FATAL" ]]; then
  if printf '%s\n' "$HEALTHY" | grep -qE "$FATAL"; then
    bad "FATAL matches a HEALTHY startup line -- every good boot would be aborted"
  else
    ok "FATAL leaves the healthy 'Free memory on device' INFO line alone"
  fi
  printf '%s\n' "$DYING" | grep -qE "$FATAL" \
    && ok "FATAL catches the two-engines-on-one-GPU failure" \
    || bad "FATAL does not catch the real out-of-memory failure"
  for sig in DEVICE_LOST OUT_OF_RESOURCES Traceback ValidationError; do
    printf 'x %s x\n' "$sig" | grep -qE "$FATAL" \
      || bad "FATAL no longer catches $sig"
  done
fi

# --- 3. the row writer must produce a row, or say so -------------------------
if need_python && [[ -n "${SKIP_DEV:-}" || 1 ]]; then
  BLOCK=$(sed -n '/^A_LABEL=/,/^row_rc=\$/p' "$ARM")
  if [[ -z "${BLOCK//[$'\n' ]/}" ]]; then
    bad "could not extract the writer block -- this test would prove nothing"
  elif ! grep -q "A_CENSUS_FILE" <<<"$BLOCK" || ! grep -q "json.dumps" <<<"$BLOCK"; then
    bad "extracted block is not the row writer"
  else
    ROWS=$(mktemp); O=$(mktemp -d)
    LABEL=CASES; ARM_PORT=8199; GATE=PASS; cgate=PASS; agate=PASS
    cg_line="15/15"; acc1="26 169 7.500"; accn="256 512 3.000"
    # the exact string that used to be handed to open() by the off-by-one
    kvtok="GPU KV cache size: 384,956 tokens"
    dtype="dtype=torch.bfloat16"; cap="Graph capturing finished;"; maxsize=80
    c1='{"label":"C1","median_tok_s_after_ttft":150.16}'
    nat='{"label":"NAT","median_tok_s_after_ttft":58.77}'
    conc='{"cells":{"conc10":{"aggregate_tok_s":224.02}}}'
    pj(){ python3 -c "import json,sys
t=sys.stdin.read(); i=t.find('{')
try: print(json.dumps(json.loads(t[i:])))
except Exception: print('null')" <<<"$1"; }
    printf '%s\n' '[w57-census] comp=norm_esimd rows=80' > "$O/CASES.census"

    eval "$BLOCK"
    if [[ $row_rc -ne 0 || ! -s "$ROWS" ]]; then
      bad "row writer produced no row (rc=$row_rc) -- the original silent failure"
    else
      python3 - "$ROWS" <<'PY' || fails=$((fails+1))
import json, sys
d = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
want = {'label':'CASES','port':8199,'gate':'PASS','correctness':'15/15',
        'kvtok':'GPU KV cache size: 384,956 tokens','ladder_max':'80'}
bad = 0
for k, v in want.items():
    if d.get(k) != v:
        print(f"  FAIL writer field {k}={d.get(k)!r} want {v!r}"); bad += 1
for k in ('c1raw','natraw','conc','census'):
    if d.get(k) in (None,'',{}):
        print(f"  FAIL writer field {k} empty"); bad += 1
if 'norm_esimd' not in d.get('census',''):
    print("  FAIL census content lost"); bad += 1
sys.exit(1 if bad else 0)
PY
      [[ $fails -eq 0 ]] && ok "row writer records every field from the values that once killed it"
    fi

    # must degrade, not vanish
    : > "$ROWS"; rm -f "$O/CASES.census"; eval "$BLOCK"
    if [[ $row_rc -eq 0 ]] && python3 -c "import json,sys;d=json.loads(open(sys.argv[1]).read());assert d['census']==''" "$ROWS" 2>/dev/null; then
      ok "missing census file degrades to empty instead of losing the row"
    else
      bad "missing census file killed the row (rc=$row_rc)"
    fi
    : > "$ROWS"; c1='Traceback (most recent call last): boom'; eval "$BLOCK"
    if [[ $row_rc -eq 0 ]] && python3 -c "import json,sys;d=json.loads(open(sys.argv[1]).read());assert d['c1raw'] is None" "$ROWS" 2>/dev/null; then
      ok "unparseable probe output yields c1raw=null, row still written"
    else
      bad "unparseable probe output lost the row (rc=$row_rc)"
    fi
    rm -f "$ROWS"; rm -rf "$O"
  fi
fi

# --- 4. rig errors must fail early, not as BOOT FAILED -----------------------
if [[ "${SKIP_DEV:-0}" != 1 ]]; then
  sed 's/RANK1_RENDER=.*/RANK1_RENDER=nonexistent-node/' "$T/devices.env" > "$T/bad-dev.env"
  out=$(DEVICES_ENV="$T/bad-dev.env" ARM_CHECK_ONLY=1 RUNSTATE_DIR="$T/runstate" bash "$ARM" CHECK 8199 2>&1); rc=$?
  if [[ $rc -eq 2 ]] && grep -q "does not exist" <<<"$out"; then
    ok "a stale device node is rejected at preflight, not as BOOT FAILED"
  else
    bad "stale device node not caught at preflight (rc=$rc): $out"
  fi

  mv "$T/runstate/dri-by-path" "$T/bp"
  out=$(ARM_CHECK_ONLY=1 DEVICES_ENV="$T/devices.env" RUNSTATE_DIR="$T/runstate" bash "$ARM" CHECK 8199 2>&1); rc=$?
  if [[ $rc -eq 2 ]] && grep -qi "by-path" <<<"$out"; then
    ok "a missing filtered by-path dir is rejected at preflight"
  else
    bad "missing by-path dir not caught at preflight (rc=$rc): $out"
  fi
  mv "$T/bp" "$T/runstate/dri-by-path"

  # and the happy path must still pass, or checks 1-4 are just "everything fails"
  rig bash "$ARM" CHECK 8199 >/dev/null 2>&1; rc=$?
  [[ $rc -eq 0 ]] && ok "a correctly-configured rig passes preflight" \
                  || bad "a correct rig fails preflight (rc=$rc) -- checks 1-4 would be vacuous"

  # 5. probe modules must be validated at PRE-FLIGHT, not only at mount time. An
  #    earlier revision printed rig OK and exited 0 before the block that checks
  #    SHA256SUMS, so a tampered or partial instrument directory passed the dry run
  #    callers use to stay safe, then failed only after production had been stopped.
  mkdir -p "$T/probes_nosum" && cp "$REPO/scripts/instrument/layernorm.py" "$T/probes_nosum/"
  cp -a "$REPO/scripts/instrument" "$T/probes_tampered"
  echo "# tampered after hashing" >> "$T/probes_tampered/layernorm.py"
  probe_rc() { rig bash "$ARM" CHECK 8199 "PROBES=$1" >/dev/null 2>&1; echo $?; }
  [[ "$(probe_rc "$T/probes_nosum")" == 2 ]] \
    && ok "a probe dir with no SHA256SUMS is rejected at preflight" \
    || bad "probe dir without SHA256SUMS passed preflight"
  [[ "$(probe_rc "$T/probes_tampered")" == 2 ]] \
    && ok "a probe dir whose modules do not match SHA256SUMS is rejected" \
    || bad "tampered probe dir passed preflight -- unverified modules could be mounted"
  [[ "$(probe_rc "/nonexistent-probe-dir")" == 2 ]] \
    && ok "a nonexistent probe dir is rejected rather than treated as uninstrumented" \
    || bad "nonexistent probe dir passed preflight"
  [[ "$(probe_rc "instrument")" == 0 ]] \
    && ok "PROBES=instrument passes preflight -- the three rejections above are not blanket" \
    || bad "PROBES=instrument rejected -- the rejections above would be meaningless"
  rig bash "$ARM" CHECK 8199 PROBES=none >/dev/null 2>&1
  [[ $? -eq 0 ]] && ok "PROBES=none stays usable (the pristine-image default)" \
                 || bad "PROBES=none rejected"

  # 6. the profile must actually supply the model paths. arm.sh used to default
  #    MODELS_DIR/TARGET_DIR/DRAFTER_DIR to whatever the author's machine had, so
  #    on any other rig a typo'd or missing profile value silently mounted a
  #    different checkpoint and recorded its throughput as the profile's. An
  #    earlier draft of this case tried to trigger the guard with a nonexistent
  #    profile, which arm.sh rejects earlier for an unrelated reason -- the case
  #    passed without ever reaching the code it claimed to test. A VALID profile
  #    that omits exactly these keys is what actually exercises it.
  mkdir -p "$REPO/profiles/w57-nofp"
  printf 'IMAGE=vllm-llmscaler:icr26.35\nPORT=8103\nMAX_NUM_SEQS=10\n' \
    > "$REPO/profiles/w57-nofp/w57-nofp.env"
  out=$(ARM_CHECK_ONLY=1 PROFILE=w57-nofp DEVICES_ENV="$T/devices.env" \
        RUNSTATE_DIR="$T/runstate" bash "$ARM" CHECK 8199 2>&1); rc=$?
  if [[ $rc -eq 2 ]] && grep -qi "refusing to guess" <<<"$out"; then
    ok "a profile that omits MODELS_DIR/TARGET_DIR/DRAFTER_DIR is rejected"
  else
    bad "profile missing model paths was not caught (rc=$rc): $out"
  fi
  rm -rf "$REPO/profiles/w57-nofp"
  # and the real profile must still clear the same guard
  rig bash "$ARM" CHECK 8199 >/dev/null 2>&1
  [[ $? -eq 0 ]] && ok "the shipped profile supplies them -- the rejection is not blanket" \
                 || bad "the real profile now fails the model-path guard"

  # 7. a misspelled knob must be refused, not silently forwarded. DTIPE=float16
  #    used to fall through to the pass-through env list, so the container booted
  #    at the profile default while the row was labelled and stamped as an fp16
  #    arm -- an inert knob recorded as a measurement. The rule is structural (only
  #    arm knobs, profile names, or VLLM_*/DISABLE_ESIMD_* are legal), so it cannot
  #    be half-blind the way an edit-distance check would be.
  typo(){ rig bash "$ARM" CHECK 8199 "$@" >/dev/null 2>&1; echo $?; }
  [[ "$(typo DTIPE=float16)" == 2 ]] \
    && ok "a misspelled DTYPE is refused rather than passed through inert" \
    || bad "DTIPE=float16 was accepted -- an inert knob can still be recorded as an arm"
  [[ "$(typo MAMBA=float16)" == 2 ]] \
    && ok "a misspelled MAMBA_SSM is refused" \
    || bad "MAMBA=float16 was accepted"
  [[ "$(typo CEILLING=80)" == 2 ]] \
    && ok "a misspelled profile variable is refused" \
    || bad "CEILLING=80 was accepted"
  # The legitimate forms must survive, or the three rejections above are just
  # "everything is refused". These are the exact shapes the W57 arms used.
  for good in "PROBES=instrument DTYPE=float16" \
              "VLLM_XPU_DISABLE_ESIMD_NORM=1" \
              "DISABLE_ESIMD_GDN_SPEC=1" \
              "NEEDLE=comp=norm_esimd" \
              "CEILING=56"; do
    [[ "$(typo $good)" == 0 ]] \
      || bad "legitimate argument rejected: $good"
  done
  ok "real knob, profile, ablation-env and needle forms all still accepted"
fi

echo
# --- 8. a measurement that did not happen must not record as a passing arm -----
# One fp16 boot came back gate=PASS with conc5/conc1 at 0 tokens, conc10 at
# 15.13 tok/s while its two sibling boots measured 237-240, and both single-stream
# probes unparseable. The only structural test then asked whether the string
# "conc10" appeared, so a skeleton of zeros passed and the row advertised itself as
# a measured arm. assert_not_degenerate now guards that path; these cases pin it
# against the REAL artifacts and against the mutations that would neuter it.
if need_python; then
  . "$HERE/../lib/degenerate.sh"
  D="$T/degen"; mkdir -p "$D"
  dg(){ printf '%s\n' "$2" > "$D/$1"; }

  # Fixtures are all written BEFORE any assertion reads them. That ordering is not
  # cosmetic: a first draft put three dg lines after the dcheck calls that consumed
  # them, and since a missing file also yields rc=1, those three cases passed while
  # testing the wrong rule -- "file absent" instead of "output unparseable".
  CELLS_OK='"conc10":{"aggregate_tok_s":240.17,"wall_s":40.1,"tokens":9728,"median_req_s":24.0},"conc5":{"aggregate_tok_s":169.16,"wall_s":30.2,"tokens":5120,"median_req_s":21.0},"conc1":{"aggregate_tok_s":46.99,"wall_s":20.5,"tokens":960,"median_req_s":21.3}'
  dg good_conc.json  "{\"test\":\"planning\",\"cells\":{$CELLS_OK}}"
  dg good_c1.json    '{"label":"C1","median_tok_s_after_ttft":160.47,"samples":10}'
  dg good_nat.json   '{"label":"NAT","median_tok_s_after_ttft":56.65,"samples":6}'
  # probe.py catches every per-iteration exception and continues, so 1 success out
  # of 10 still prints a clean median. That is a partial measurement, not a result.
  dg few_c1.json     '{"label":"C1","median_tok_s_after_ttft":160.47,"samples":1}'
  dg null_c1.json    'null'
  dg empty_nat.json  ''
  dg nullmed_c1.json '{"label":"C1","median_tok_s_after_ttft":null,"n":10}'
  # A raw stdout capture is not necessarily ONLY the JSON. conc.py prints its
  # WARMUP_FAILED line to stdout, and any interpreter-shutdown warning lands there
  # too. The check must find the object rather than demand a pure file, or a stray
  # diagnostic rejects a healthy boot -- the expensive direction of failure.
  dg stray_conc.json "Exception ignored in: <urllib3.connectionpool>  ResourceWarning"
  printf '%s\n' "{\"test\":\"planning\",\"cells\":{$CELLS_OK}}" >> "$D/stray_conc.json"
  # verbatim from FP16_PROBED2: two cells completed nothing
  dg bad_conc.json   '{"test":null,"cells":{"conc10":{"aggregate_tok_s":15.13,"wall_s":321.4,"tokens":4864,"median_req_s":10.72},"conc5":{"aggregate_tok_s":0.0,"wall_s":0.0,"tokens":0,"median_req_s":null},"conc1":{"aggregate_tok_s":0.0,"wall_s":0.0,"tokens":0,"median_req_s":null}}}'

  # Each rejection case also pins the REASON, not just the exit code. Without the
  # reason match, every probe case would pass on a missing file and the whole block
  # would be green against a check that tested nothing.
  #
  # First, the extractor itself. A previous revision replaced `tr -d '\n'` with
  # `grep -o '{.*}'` and EVERY arm would have been rejected as unparseable, because
  # probe.py prints INDENTED json and a single-line brace match returns the lone
  # character "{". Nothing here caught it, because every fixture below is written as
  # one-line JSON -- so the harness stayed green against a pipeline that could never
  # parse real probe output. These cases use the multi-line form probe.py emits.
  printf '{\n  "label": "C1",\n  "median_tok_s_after_ttft": 160.47,\n  "samples": 10\n}\n' \
    > "$D/pretty_c1.json"
  jo(){ # -> prints "obj" or "none"; rc 0 when the file holds a decodable object
    python3 -c "
import sys
sys.path.insert(0, sys.argv[1])
from jsonobj import jsonobj
o = jsonobj(sys.argv[2])
print('obj' if o is not None else 'none')
sys.exit(0 if o is not None else 1)" "$HERE/../bench" "$1"
  }
  # probe.py's real indented output, a stray line ahead of a good payload, and the
  # conc fixture above are all accepted; a file with no object is refused.
  jo "$D/pretty_c1.json" >/dev/null \
    && ok "jsonobj decodes probe.py's indented output" \
    || bad "jsonobj rejected indented JSON -- every arm would be discarded as unparseable"
  jo "$D/stray_conc.json" >/dev/null \
    && ok "jsonobj skips a stray line before the payload" \
    || bad "jsonobj rejected a payload preceded by a diagnostic line"
  printf 'no json at all here\n' > "$D/nojson.txt"
  jo "$D/nojson.txt" >/dev/null \
    && bad "jsonobj matched an object in a file containing none" \
    || ok "jsonobj returns None when there is no object"
  # A broken parser import must say so, not exit 1 in silence. Silent failure looks
  # exactly like a degenerate measurement, and in this very file it would look like a
  # mutation that proved nothing.
  mkdir -p "$D/emptydir"
  out=$(DG_BENCH_DIR="$D/emptydir" assert_not_degenerate "$D/good_conc.json" \
        "$D/good_c1.json" "$D/good_nat.json" 2>&1)
  grep -q 'cannot import jsonobj' <<<"$out" \
    && ok "a missing parser fails loudly instead of reading as degenerate" \
    || bad "broken import was silent or mislabelled: $out"
  dcheck(){ # $1=label $2=want-rc $3=reason-regex $4=conc $5=c1 $6=nat [$7=c1-need $8=nat-need]
    local out rc want=$2 re=$3
    out=$(A_C1_NEED=${7:-} A_NAT_NEED=${8:-} \
            assert_not_degenerate "$D/$4" "$D/$5" "$D/$6" 2>&1); rc=$?
    if [[ $rc -ne $want ]]; then
      bad "$1 -> rc=$rc, wanted $want ($out)"
    elif [[ -n $re ]] && ! grep -qE "$re" <<<"$out"; then
      bad "$1 -> right rc, wrong reason: $out (wanted /$re/)"
    else
      ok "$1 -> rc=$rc ${out:+($out)}"
    fi
  }
  dcheck "healthy arm accepted"                  0 ''            good_conc.json  good_c1.json  good_nat.json
  dcheck "real FP16_PROBED2 conc rejected"       1 'completed 0 requests' bad_conc.json good_c1.json good_nat.json
  dcheck "unparseable C1 probe rejected"         1 'C1 probe did not parse' good_conc.json null_c1.json good_nat.json
  dcheck "empty NAT probe rejected"              1 'NAT probe did not parse' good_conc.json good_c1.json empty_nat.json
  dcheck "null-median probe rejected"            1 'no positive median' good_conc.json nullmed_c1.json good_nat.json
  dcheck "missing conc file rejected"            1 'conc json unreadable' nope.json good_c1.json good_nat.json
  # A diagnostic line before the payload must NOT reject the arm: the payload is
  # still there and complete, so rejecting it discards a healthy boot.
  dcheck "stray line before conc json accepted"  0 ''            stray_conc.json good_c1.json good_nat.json
  # A median over one surviving sample is not a measurement: probe.py swallows
  # per-iteration exceptions, so 1-of-10 prints a clean number. This is the case
  # that the median-only check let through.
  dcheck "1-of-10 C1 samples rejected"           1 'C1 probe completed 1/10' good_conc.json few_c1.json good_nat.json 10 6
  # ...and the same artifact must be ACCEPTED when no expectation is supplied, so
  # the case above proves it was the rule firing and not an unrelated rejection.
  dcheck "same file ok with no expectation"      0 ''            good_conc.json  few_c1.json  good_nat.json

  # Mutations, one per rule, each blanking the LOOP ITSELF rather than a single
  # branch. Blinding just `if not cell.get('tokens')` is not a valid mutation: the
  # `elif` still fires on a zero-token / zero-tok/s cell, so the check would stay
  # strict and the mutation would fail for a reason that has nothing to do with the
  # rule under test. An empty iterable is the only edit that isolates one rule.
  mcheck(){ # $1=label $2=sed-expr $3=conc $4=c1 $5=nat [$6=c1-need $7=nat-need]
    sed "$2" "$HERE/../lib/degenerate.sh" > "$D/mut.sh"
    # Guard against the mutation never applying: a sed pattern that matches nothing
    # produces an unchanged copy, which then rejects exactly like the real code and
    # reads as "the rule is not load-bearing". Assert the file actually changed.
    if cmp -s "$D/mut.sh" "$HERE/../lib/degenerate.sh"; then
      bad "MUTATION $1: sed matched nothing -- the mutant is the original, so this verdict is meaningless"
      return
    fi
    # the mutant must now ACCEPT an artifact the real check rejects, which is what
    # proves the assertion above depends on this specific rule
    # The mutant env MUST be exported with explicit `export`, on its own statement,
    # not written as a prefix on the `.` line. Two separate traps:
    #  * `X=v . file` does NOT export X. Bash treats an assignment preceding the
    #    source builtin as a plain shell assignment and rolls it back when the
    #    source returns, so assert_not_degenerate never saw DG_BENCH_DIR, died on an
    #    ImportError, and every mutation read as "still rejected" -- which this
    #    harness would then report as "this rule is not load-bearing", a verdict
    #    caused entirely by a harness path bug rather than the mutation.
    #  * `A=x B=y cmd` is a LIST: only the final word is the command, earlier
    #    assignments set the caller's vars without exporting them either.
    # The mutant also lives in $T with no sibling bench/, so DG_BENCH_DIR is required.
    if ( export A_C1_NEED=${6:-} A_NAT_NEED=${7:-} DG_BENCH_DIR="$HERE/../bench"
         . "$D/mut.sh"
         assert_not_degenerate "$D/$3" "$D/$4" "$D/$5" >/dev/null 2>&1 ); then
      ok "MUTATION $1: check goes blind without this rule -- the case above is load-bearing"
    else
      bad "MUTATION $1: still rejected -- the case above does not depend on the rule it claims to test"
    fi
  }
  mcheck "cell loop emptied" "s/for name in ('conc10', 'conc5', 'conc1'):/for name in ():/" \
         bad_conc.json good_c1.json good_nat.json
  mcheck "probe loop emptied" "s/for tag, key, need in (('C1', 'A_C1RAW_F', 'A_C1_NEED'), ('NAT', 'A_NATRAW_F', 'A_NAT_NEED')):/for tag, key, need in ():/" \
         good_conc.json null_c1.json good_nat.json
  mcheck "samples rule emptied" "s/if want and (d.get('samples') or 0) < want:/if False:/" \
         good_conc.json few_c1.json good_nat.json 10 6
fi

echo
if [[ $fails -eq 0 ]]; then
  echo "falsify/arm-cases: PASS -- arm.sh overrides, fatal pattern, row writer, preflight guards, and the degeneracy gate can all fail"
  exit 0
fi
echo "falsify/arm-cases: FAIL ($fails)"
exit 1
