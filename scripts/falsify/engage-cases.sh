#!/usr/bin/env bash
# Falsification harness for scripts/lib/engage.sh.
#
# engage.sh decides whether an arm's dtype knob actually reached the engine, so a
# silent failure there turns a rig bug into a published number under the wrong
# label. Every case runs both directions: the assertion must pass on evidence and
# fail without it, and each rule carries a mutation showing the case depends on it.
#
# Fixtures are generated rather than read from a live rig, so the harness runs
# anywhere. The synthetic banner reproduces the two shapes the real engine emits
# (verified against the recorded W57 logs: a single dtype=torch.* token on the
# "Initializing a V1 LLM engine" line, and the Qwen3.5 mamba_ssm_dtype warning
# only when the passed value disagrees with config.json). When a real arm log
# directory is available, LOGS_DIR adds the real-log cases on top.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/../lib/engage.sh"

fails=0
ok(){ echo "  ok   $*"; }
bad(){ echo "  FAIL $*"; fails=$((fails+1)); }
chk(){ # label want-rc fn log arg
  local label=$1 want=$2 fn=$3 log=$4 arg=$5 out rc
  out=$($fn "$log" "$arg"); rc=$?
  if [[ $rc -eq $want ]]; then ok "$label -> rc=$rc ($out)"
  else bad "$label -> rc=$rc, wanted $want ($out)"; fi
}

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

# --- fixtures ---------------------------------------------------------------
# A bf16 control banner: one dtype token, no mamba warning (the pin matched).
cat > "$T/bf16.log" <<'EOF'
INFO [core.py:116] Initializing a V1 LLM engine (v0.26.1.dev0+g568afb3a1) with config: model='/models/target', speculative_config=SpeculativeConfig(method='dspark', model='/models/drafter', num_spec_tokens=7), tokenizer='/models/target', dtype=torch.bfloat16, max_seq_len=262144, quantization=fp8, kv_cache_dtype=fp8, device_config=xpu
INFO [cache.py] GPU KV cache size: 384,956 tokens
EOF
# A float16 arm banner plus the disagreement warning the engine emits for
# --mamba-ssm-cache-dtype=float16 against a float32 checkpoint.
cat > "$T/fp16.log" <<'EOF'
INFO [core.py:116] Initializing a V1 LLM engine (v0.26.1.dev0+g568afb3a1) with config: model='/models/target', speculative_config=SpeculativeConfig(method='dspark', model='/models/drafter', num_spec_tokens=7), tokenizer='/models/target', dtype=torch.float16, max_seq_len=262144, quantization=fp8, kv_cache_dtype=fp8, device_config=xpu
WARNING [config.py:770] Qwen3.5 model specifies mamba_ssm_dtype='float32' in its config, but --mamba-ssm-cache-dtype='float16' was passed. Using the user-specified value.
INFO [cache.py] GPU KV cache size: 414,646 tokens
EOF
printf 'x dtype=torch.float16 y dtype=torch.bfloat16 z\n' > "$T/two.log"
printf 'no dtype token here\n' > "$T/none.log"
: > "$T/empty.log"

echo "=== dtype: must pass on evidence and fail without it ==="
chk "bf16 log at bfloat16       " 0 assert_dtype_applied "$T/bf16.log" bfloat16
chk "fp16 log at float16        " 0 assert_dtype_applied "$T/fp16.log" float16
chk "bf16 log claimed float16   " 1 assert_dtype_applied "$T/bf16.log" float16
chk "fp16 log claimed bfloat16  " 1 assert_dtype_applied "$T/fp16.log" bfloat16
chk "auto is not a match target " 1 assert_dtype_applied "$T/fp16.log" AUTO
chk "two tokens -> ambiguous    " 1 assert_dtype_applied "$T/two.log" float16
chk "no token                   " 1 assert_dtype_applied "$T/none.log" float16
chk "empty file                 " 1 assert_dtype_applied "$T/empty.log" float16
chk "missing file               " 1 assert_dtype_applied "$T/nope.log" float16
chk "unpinned -> skip, no check " 0 assert_dtype_applied "$T/none.log" ""
chk "auto pin -> skip           " 0 assert_dtype_applied "$T/none.log" auto

echo
echo "=== ssm: the engine warning is the only available signal ==="
chk "fp16 log, float16 provable " 0 assert_ssm_applied "$T/fp16.log" float16
chk "bf16 log, float16 absent   " 1 assert_ssm_applied "$T/bf16.log" float16
chk "float32 == checkpoint, rc2 " 2 assert_ssm_applied "$T/fp16.log" float32
chk "unpinned -> skip           " 0 assert_ssm_applied "$T/empty.log" ""
chk "auto pin -> skip           " 0 assert_ssm_applied "$T/empty.log" auto
chk "empty file                 " 1 assert_ssm_applied "$T/empty.log" float16

echo
echo "=== mutations: each rule must be load-bearing ==="
# The mutant must run in a subshell (sourcing it in the parent would overwrite the
# real functions for every later case), but that also means ok/bad inside it would
# increment `fails` in the subshell and lose it -- which is exactly how an earlier
# draft of this file reported PASS while printing a FAIL line. So the subshell only
# reports the mutant's verdict through its exit status, and the parent decides.
mut(){ # label sed-expr fn log arg
  local label=$1 expr=$2 fn=$3 log=$4 arg=$5
  sed "$expr" "$HERE/../lib/engage.sh" > "$T/eng.sh"
  if ( . "$T/eng.sh"; "$fn" "$log" "$arg" >/dev/null 2>&1 ); then
    ok "MUTATION $label: goes blind without this rule"
  else
    bad "MUTATION $label: still rejected -- the case above does not depend on this rule"
  fi
}
mut "dtype comparison bypassed " 's/if \[\[ "\$have" != "dtype=torch.\$want" \]\]; then/if false; then/' \
    assert_dtype_applied "$T/bf16.log" float16
mut "unprovable claimed as ok  " 's/^    return 2$/    return 0/' \
    assert_ssm_applied "$T/fp16.log" float32
# The empty-file guard is DEFENSIVE too: removing it lets the count check reject
# the same file anyway, so it is not verdict-changing. Pin that its message is the
# one a reader actually gets, so the branch cannot be deleted silently.
sed 's/^    echo "assert_dtype_applied: log \$L is empty or missing"; return 1$/    :/' \
    "$HERE/../lib/engage.sh" > "$T/eng3.sh"
real_msg=$(assert_dtype_applied "$T/empty.log" float16)
if ( . "$T/eng3.sh"; assert_dtype_applied "$T/empty.log" float16 >/dev/null 2>&1 ); then
  bad "MUTATION empty-file guard: the file was accepted"
else
  mut_msg=$( . "$T/eng3.sh"; assert_dtype_applied "$T/empty.log" float16 )
  if grep -q "empty or missing" <<<"$real_msg" && ! grep -q "empty or missing" <<<"$mut_msg"; then
    ok "MUTATION empty-file guard: the empty-vs-no-tokens distinction disappears, so the branch is exercised"
  else
    bad "MUTATION empty-file guard: real='$real_msg' mutant='$mut_msg'"
  fi
fi
# The one-token count guard is DEFENSIVE: with it removed the string comparison
# still rejects a two-token banner, so an accept/reject mutation is meaningless.
# Pin instead that the branch is exercised -- a multi-token banner must be
# reported as ambiguous, not as a plain mismatch, or a future edit can delete it
# without any case noticing.
sed 's/if \[\[ "\$n" != 1 \]\]; then/if false; then/' "$HERE/../lib/engage.sh" > "$T/eng2.sh"
real_msg=$(assert_dtype_applied "$T/two.log" float16)
( . "$T/eng2.sh"
  mut_msg=$(assert_dtype_applied "$T/two.log" float16)
  if grep -q "expected exactly one" <<<"$real_msg" \
     && ! grep -q "expected exactly one" <<<"$mut_msg"; then
    echo "  ok   MUTATION count guard: the ambiguous-vs-mismatch distinction disappears, so the branch is exercised"
  else
    echo "  FAIL MUTATION count guard: real='$real_msg' mutant='$mut_msg'"; exit 1
  fi ) || fails=$((fails+1))

# --- opportunistic real-log cases ------------------------------------------
LOGS=${LOGS_DIR:-}
if [[ -n "$LOGS" && -s "$LOGS/FP16_PROBED.engine.log" && -s "$LOGS/CALIB_PROBED.engine.log" ]]; then
  echo
  echo "=== real W57 logs: the shipped arms must satisfy their own assertions ==="
  chk "real control at bfloat16 " 0 assert_dtype_applied "$LOGS/CALIB_PROBED.engine.log" bfloat16
  chk "real control claimed fp16" 1 assert_dtype_applied "$LOGS/CALIB_PROBED.engine.log" float16
  chk "real fp16 at float16     " 0 assert_dtype_applied "$LOGS/FP16_PROBED.engine.log" float16
  chk "real fp16 claimed bf16   " 1 assert_dtype_applied "$LOGS/FP16_PROBED.engine.log" bfloat16
  chk "real fp16 ssm provable   " 0 assert_ssm_applied   "$LOGS/FP16_PROBED.engine.log" float16
  chk "real control ssm absent  " 1 assert_ssm_applied   "$LOGS/CALIB_PROBED.engine.log" float16
else
  echo
  echo "  SKIP real-log cases (set LOGS_DIR to a directory with *.engine.log)"
fi

echo
[[ $fails -eq 0 ]] && { echo "falsify/engage-cases: PASS -- dtype and ssm engagement can both fail to be proven"; exit 0; }
echo "falsify/engage-cases: FAIL ($fails)"; exit 1
