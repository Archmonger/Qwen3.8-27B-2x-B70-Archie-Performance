#!/usr/bin/env bash
# Knob-application assertions for scripts/bench/arm.sh, shared with
# scripts/falsify/arm-cases.sh.
#
# Usage:  source scripts/lib/engage.sh
#         assert_dtype_applied <engine-log> <want-dtype>      # "" / auto = skip
#         assert_ssm_applied   <engine-log> <want-ssm>         # "" / auto = skip
#           -> print one line on stdout, return 0 when the knob provably reached
#              the engine, 1 when it did not.
#
# Why this is separate from attest.sh: attest covers ceiling, ladder, graphs,
# spec k, method, schedule, capture, V2 runner and KV dtype -- but NOT the model
# dtype. Without an assertion, an arm asked to boot at float16 that silently came
# up at bfloat16 would record gate=PASS under an fp16 label with entirely
# plausible numbers. That is the one rig bug that converts directly into a wrong
# published result, because everything downstream trusts the label.
#
# Evidence rules, and why each is the shape it is:
#   * The engine banner ("Initializing a V1 LLM engine ... with config: ...")
#     prints the RESOLVED target dtype as a single `dtype=torch.X` token, which is
#     why auto is not accepted as a match target: it never appears in the log.
#     The drafter's dtype is not printed on that line, so there is exactly one
#     candidate token; the assertion requires exactly one and rejects zero or
#     several rather than guessing with `tail -1`.
#   * mamba_ssm_cache_dtype is NOT in the banner. The only signal the engine
#     emits is the warning Qwen3_5ForConditionalGenerationConfig
#     (model_executor/models/config.py:752-777) logs when the passed value
#     disagrees with the checkpoint's mamba_ssm_dtype. For this checkpoint that is
#     float32, so any non-float32 pin produces the warning and float32 produces
#     nothing. A float32 pin therefore returns 2 = "cannot be proven here" rather
#     than 0, so callers cannot mistake an unprovable claim for a verified one.

assert_dtype_applied(){
  local L=${1:?engine log} want=${2:-}
  [[ -n "$want" && "$want" != auto ]] || return 0
  if [[ ! -s "$L" ]]; then
    echo "assert_dtype_applied: log $L is empty or missing"; return 1
  fi
  local hits n have
  # grep -o emits one line per match; grep -c would count LINES, and the whole
  # banner is one line, so -c would report 1 however many tokens it contains.
  hits=$(grep -oE 'dtype=torch\.[a-z0-9]+' "$L" || true)
  n=$(printf '%s' "$hits" | grep -c . || true)
  if [[ "$n" != 1 ]]; then
    echo "assert_dtype_applied: expected exactly one dtype=torch.* token, found ${n:-0} -- cannot tell which config it came from"
    return 1
  fi
  have=$hits
  if [[ "$have" != "dtype=torch.$want" ]]; then
    echo "assert_dtype_applied: asked DTYPE=$want but the engine reports $have"
    return 1
  fi
  echo "dtype engagement ok ($have)"
  return 0
}

assert_ssm_applied(){
  local L=${1:?engine log} want=${2:-}
  [[ -n "$want" && "$want" != auto ]] || return 0
  if [[ ! -s "$L" ]]; then
    echo "assert_ssm_applied: log $L is empty or missing"; return 1
  fi
  # The checkpoint pins float32, and the engine only warns on disagreement.
  if [[ "$want" == float32 ]]; then
    echo "assert_ssm_applied: SKIPPED -- float32 equals the checkpoint's mamba_ssm_dtype, so the engine emits nothing to detect (unproven, not verified)"
    return 2
  fi
  if ! grep -q "mamba-ssm-cache-dtype='$want' was passed" "$L"; then
    echo "assert_ssm_applied: --mamba-ssm-cache-dtype=$want never reached the engine (no config warning emitted)"
    return 1
  fi
  echo "ssm engagement ok ($want)"
  return 0
}
