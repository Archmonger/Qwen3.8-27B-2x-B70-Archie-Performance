#!/usr/bin/env bash
# In-container vLLM launcher for the intel/llm-scaler image with DSpark
# speculative decoding on XPU. Referenced by every profile's compose file as
# /serve.sh; the profile env file supplies the values, this file assembles the
# `vllm serve` command line.
#
# Behaviour here is what the measured run executed. Comments carry the reasons,
# because most of these knobs were argued about and several changed their verdict
# only after measurement. The defect references point into docs/.
set -euo pipefail

# The image entrypoint normally does this; we override the entrypoint so do it
# here. oneAPI vars.sh references unbound vars, so relax nounset while sourcing.
# Nothing is loaded from the host's /opt/intel/oneapi -- the venv bundles it.
set +u
source /opt/intel/oneapi/setvars.sh --force > /dev/null
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force > /dev/null
set -u

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# --- XPU graphs.
#     Off by default because graphs are only safe with DSpark in one specific
#     shape; see docs/cudagraph-stride-defect.md. Opt in with XPU_GRAPH=1 AND a
#     capped --compilation-config in EXTRA_VLLM_ARGS. Both halves are required:
#     with the image's default cudagraph_capture_sizes (up to 96) capture faults
#     here with UR_RESULT_ERROR_DEVICE_LOST / OUT_OF_RESOURCES for both
#     FULL_AND_PIECEWISE and FULL_DECODE_ONLY, so an uncapped graph request does
#     not boot. The deployed profile pairs this with
#       {"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY",
#        "max_cudagraph_capture_size":MAX_NUM_SEQS x (1+k)}
#     and static k=7, which is the combination measured correct (30/30) and worth
#     227.13 tok/s aggregate at the MAX_NUM_SEQS cap.
#     Earlier notes in this file claimed graphs were "not useful with DSpark" and
#     that every compile variant lost to eager. Both were superseded: graphs
#     without DSpark are ~2x, and graphs WITH dspark are correct and fastest at a
#     consistent stride. What still holds is that torch.compile (as opposed to
#     graphs) collapses DSpark acceptance to ~1.00, which is why mode is "none".
if [[ "${XPU_GRAPH:-0}" == "1" ]]; then
  export VLLM_XPU_ENABLE_XPU_GRAPH=1
  # The sampler kernel is bundled with the graph switch for historical
  # reasons; XPU_SAMPLER=0 separates them so a graph run can be attributed
  # between capture and sampler. Swept: neither this nor ZE_FLAT_DEVICE_HIERARCHY
  # improves on baseline. Kept as switches, not as tuning.
  if [[ "${XPU_SAMPLER:-1}" == "1" ]]; then
    export VLLM_XPU_USE_SAMPLER_KERNEL=1
  fi
fi

# --- oneCCL / level-zero topology.
#     Same exchange/topology settings bare metal used (pidfd + P2P + SYCL
#     kernels), and they work here. Inside the container the exchange mode is
#     irrelevant until /dev/dri/by-path is filtered (docs/dri-by-path.md) -- oneCCL
#     aborts in fill_device_fds on the dangling entries before CCL tuning ever
#     matters. Host interface pinning (CCL_ATL_*/FI_TCP_IFACE) is dropped below.
export CCL_TOPO_P2P_ACCESS="${CCL_TOPO_P2P_ACCESS:-1}"
export CCL_ZE_IPC_EXCHANGE="${CCL_ZE_IPC_EXCHANGE:-pidfd}"
export CCL_ENABLE_SYCL_KERNELS="${CCL_ENABLE_SYCL_KERNELS:-1}"
# Pinned 0 deliberately: on Battlemage, handing oneCCL a tmp buffer for allgatherv
# faults. It is the opposite of the allreduce setting on the next line on purpose --
# do not "harmonise" the two.
export CCL_SYCL_ALLGATHERV_TMP_BUF=0
# The ONLY mitigation found for the oneCCL IPC-handle wedge: TP2 workers can spin in
# ze_handle_manager during the handle exchange, generating nothing while /health
# keeps returning 200, and it does not self-heal. Measured 0/6 wedges with this set
# vs 4/5 same-run controls (p=0.0152). Reproduce with scripts/soak.sh.
export CCL_SYCL_ALLREDUCE_TMP_BUF="${CCL_SYCL_ALLREDUCE_TMP_BUF:-1}"
export CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072
export CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-COMPOSITE}"
export TRITON_DEFAULT_BACKEND=intel

# dspark exists only in the V2 GPU model runner. Pinning it explicitly also keeps a
# no-spec comparison arm on the same runner: config/vllm.py forces V2 when
# method=dspark but leaves a no-spec run of this architecture free to default to V1,
# so without this an A/B measures V2-vs-V1 as much as spec-vs-none.
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"

# FP8 checkpoints: convert weights with CPU offload so the ~29 GB of fp8 shards
# do not exhaust device memory during load.
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT="${VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT:-1}"

# Recorded dead end -- do not expect CCL_NO_P2P=1 to work on this hardware.
#
# A public replicator of a similar deployment needed a no-P2P oneCCL path because
# *his* host hung during compilation on CCL_TOPO_P2P_ACCESS=1 + drmfd, and
# privileged whole-DRI access did not help. His block pins `lo`, which does exist
# inside a container, so it was worth testing rather than dismissing on "host
# interface names" grounds.
#
# Measured here: it cannot start at all. Every variant reaches
#   oneCCL: ze_handle_manager.cpp:42 mem_to_ipc_handle: EXCEPTION:
#   device_fd is invalid value
# under CCL_ZE_IPC_EXCHANGE=pidfd, under sockets, and with
# ZE_FLAT_DEVICE_HIERARCHY=FLAT as well as COMPOSITE. The identical block on the
# original author's own image fails the same way, so it is this container's
# device-fd situation, not the llm-scaler image. His no-P2P requirement came from a
# PCIe topology this box does not share, and TP2 collectives are already cheap here
# (TP=1 no-spec INT4 10.13 tok/s vs TP2 8.60), which bounds the whole lever at
# roughly 1.5 tok/s even hypothetically.
#
# Kept only as a switch to reproduce the failure; the default P2P path below is the
# one that works. See docs/oneccl-and-p2p.md.
if [[ "${CCL_NO_P2P:-0}" == "1" ]]; then
  export FI_PROVIDER="${FI_PROVIDER:-tcp}"
  export FI_TCP_IFACE="${FI_TCP_IFACE:-lo}"
  export CCL_ATL_TRANSPORT=ofi
  export CCL_SEND=direct
  export CCL_RECV=direct
  export CCL_TOPO_P2P_ACCESS=0
  export CCL_ZE_IPC_EXCHANGE="${CCL_ZE_IPC_EXCHANGE:-pidfd}"
else
  unset CCL_ATL_TRANSPORT CCL_ATL_SRC_IFACE FI_TCP_IFACE 2>/dev/null || true
fi

MODEL="${MODEL:-/models/target}"
DRAFTER="${DRAFTER:-/models/drafter}"
PORT="${PORT:-8103}"
SERVED_NAME="${SERVED_NAME:-secondary}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
TP_SIZE="${TP_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
# Off by default, which is the safe thing to ship rather than a verdict on the tier:
# it is not a performance knob on this deployment (it never serves a read --
# kv_offload_cpu_cache_read_usage_perc and the external prefix-cache hit rate both
# stay 0.0 across every boot measured) and it consumes host RAM proportional to use,
# so a launcher-wide default of "on" would cost RAM and buy nothing. Nothing in
# this repository sets it non-zero: profiles/fp8/fp8.env ships 0 too, and the one
# box that wants the capacity sets it in the gitignored profiles/fp8/host.env,
# which compose bakes into this container's environment at create. By the time
# this line runs we are inside that container and cannot see the overlay -- the
# value simply arrives, which is why the default here is a backstop and not the
# thing anyone should expect to be relying on.
#
# A non-zero value here is only safe on an image carrying the vllm#52923 KV-offload
# hotfix (image/patches/); without it, a single long request kills EngineCore with an
# AssertionError in offloading/scheduler.py. setup.sh attests the pairing and refuses
# to start an unpatched image with the tier on. See docs/kv-offload-defect.md.
KV_OFFLOAD_SIZE="${KV_OFFLOAD_SIZE:-0}"

# --- Speculative decoding.
#
# SPEC_METHOD=dspark is the default and the only method that boots with the
# published drafter on 0.26: dspark is V2-runner-only and natively performs the
# "anchor is the first prediction" readout that other routes hand-patch into
# dflash.py, so no image patch is needed. method=dflash auto-prefixes to an
# unregistered architecture and dies at SpeculativeConfig validation before any
# model load. See docs/dspark-drafter.md.
#
# NUM_SPEC_TOKENS (k) is NOT a tuning knob when graphs are on. A FULL decode graph
# replays another position's logits unless every captured stride equals 1+k, and the
# failure is invisible to throughput and acceptance metrics:
#   graphs ON  k=7 -> 30/30 correct      k=6 -> 30/30   k=1 -> 30/30
#   graphs ON  k=4 ->  6/15 CORRUPT
#   graphs OFF k=4 -> 30/30 correct   (so the drafter alone is fine)
# There is no parity or divisibility rule that explains that pattern, so k=7 is a
# validated pin rather than an optimum. k != 7 also costs ~30% of single-stream
# speed (k=6/k=4 C1 ~101-105 vs k=7's 151).
#
# The default below is 7 -- the safe shape. (The file this launcher was derived
# from defaulted to 4, which is the value that corrupts once graphs are enabled;
# it was always pinned by the caller, so the change affects no measured run.)
# If you set XPU_GRAPH=1, keep k=7 and re-check it against
# docs/cudagraph-stride-defect.md before changing anything else.
#
# SPEC_TOKENS_SCHEDULE (vLLM's native dynamic speculative decoding: a JSON list of
# inclusive [range_start, range_end, num_speculative_tokens] triples) must NOT be
# set with graphs on. The drafter always drafts the scalar k while the capture loop
# takes its strides from this schedule, so a graph captured at a stride the drafter
# never produces replays another position's logits -- measured 6-13/30 correct with
# acceptance counters identical to a healthy run. setup.sh aborts if it is set.
SPEC_ARGS=()
if [[ "${SPEC:-1}" == "1" ]]; then
  SPEC_JSON="{\"method\":\"${SPEC_METHOD:-dspark}\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS:-7}"
  if [[ -n "${SPEC_TOKENS_SCHEDULE:-}" ]]; then
    SPEC_JSON="${SPEC_JSON},\"num_speculative_tokens_per_batch_size\":${SPEC_TOKENS_SCHEDULE}"
  fi
  SPEC_ARGS=(--speculative-config "${SPEC_JSON}}")
fi

# --- Compilation mode.
# COMPILE=0 adds --enforce-eager. The deployed profile sets COMPILE=1 and gets its
# speed from GRAPHS, not from torch.compile: compiling the base model does make it
# faster in isolation (C1 probe, spec off: eager 9.98 -> compile 15.01 ->
# compile+capped-graphs 22.49) but collapses DSpark acceptance to ~1.00, so every
# compiled variant loses once drafting is on. That is why EXTRA_VLLM_ARGS pins
# --compilation-config mode "none" while still enabling FULL_DECODE_ONLY.
# torch.compile itself runs clean once the Dockerfile driver overlay is in place, so
# this is a spec-decode interaction, not a broken toolchain.
EXTRA_ARGS=()
if [[ "${COMPILE:-0}" != "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "${PREFIX_CACHE:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-prefix-caching)
else
  EXTRA_ARGS+=(--no-enable-prefix-caching)
fi

# KV_OFFLOAD_SIZE=0 omits --kv-offloading-size entirely; handing vLLM a literal
# 0 is not the same as omitting the flag.
KV_ARGS=()
# Test the resolved value from the assignment above, not a fresh ${VAR:-50}: the
# two could disagree, and an empty value then passed the guard (empty -> 50) while
# passing "$VAR" handed the parser a flag with no argument.
if [[ "$KV_OFFLOAD_SIZE" != "0" ]]; then
  KV_ARGS+=(--kv-offloading-size "$KV_OFFLOAD_SIZE")
fi

# --- activation / GDN-state dtype. W57 measured both knobs on this host against
#     matched control boots. They are two different levers: DTYPE buys speed,
#     MAMBA_SSM_CACHE_DTYPE buys KV capacity. Both default to float16 because
#     this repo's rule is that the launcher default and the published numbers are
#     the most optimized measured permutation, not the historical one. `auto`
#     (what the launcher shipped with before this was measured) remains the
#     documented control spelling: every A/B against the old baseline passes
#     DTYPE=auto explicitly through scripts/bench/arm.sh.
#
# Measured (paired A/B on one rig via scripts/bench/arm.sh, shipped harnesses,
# PROBES=instrument on both arms so neither pays probe overhead alone; two
# controls and two candidates, order alternated, all gates PASS):
#
#   label               C1 probe   conc=1   conc=5  conc=10  GPU KV tokens
#   DTYPE=auto             151.93    43.97   155.17   218.65        384,956
#   DTYPE=auto             151.19    44.07   150.76   223.52        384,956
#   DTYPE=float16          160.47    46.99   169.16   240.17        414,646
#   DTYPE=float16          160.74    47.07   171.13   237.23        414,646
#   control mean           151.56    44.02   152.96   221.09
#   candidate mean         160.61    47.03   170.14   238.70
#   delta                   +6.0%    +6.8%   +11.2%     +8.0%          +7.7%
#
# Within-arm spread is 0.27-0.74 on C1 and 2.0-4.9 on the conc cells, so every
# delta above is several times the spread. Quote them with the ranges, not alone.
#
# The attribution split (earlier run, same rig, same harness, probes mounted on
# every arm; one launch per label except where a rep is shown. Its KV totals
# differ from the table above by a few hundred tokens, ordinary boot-to-boot
# variation) is what justifies defaulting BOTH knobs:
#
#   label        knobs                    conc=10          conc=5   GPU KV tokens
#   CALIB2/3     defaults                224.18/224.02    156.37/155.31      384,956
#   MAMBA16      ssm=float16 only            225.79         156.49           397,616
#   FP16ONLY     DTYPE=float16 only     244.76/241.53    161.86/168.96       389,265
#   FP16FULL     both          237.66/245.08/242.86     166.04/171.45        415,029
#
# DTYPE is where the speed is: ssm alone is 225.79 against a 224.10 control mean,
# while DTYPE alone is 243.15 and both together 241.87. The two fp16 rows sit
# inside each other's spread, so the ssm pin buys nothing on throughput and is
# defaulted in for capacity only. Its capacity is also NOT additive -- +12,660
# tokens alone, +30,073 alongside DTYPE, against a 384,956 control -- because the
# pool is whatever memory is left divided by a per-token page size that both
# dtypes move. Do not quote a per-knob token delta as if the two summed.
#
# WHY IT IS FASTER IS NOT ESTABLISHED, and an earlier revision of this comment was
# wrong about it. It asserted the win came from two XPU fast paths gated on
# float16 that are dead under bfloat16. Both are real paths, and both were
# confirmed absent under bf16 by a census probe -- but ablation refuted the causal
# claim. Each was switched off in turn, and both off at once, with the dtype pins
# still set and the census proving the ablation applied:
#
#   arm                                    ESIMD norm  GDN spec   conc10   C1
#   DTYPE=float16 (candidate)                present      present  240.17   160.47
#   DTYPE=float16 (candidate, rep 2)         present      present  237.23   160.74
#   + VLLM_XPU_DISABLE_ESIMD_NORM=1             OFF       present  240.84   161.36
#   + DISABLE_ESIMD_GDN_SPEC=1               present         OFF   243.41   160.45
#   + both                                      OFF          OFF   239.99   160.05
#
# The win does not move when either kernel is off, or when both are. So neither
# ESIMD kernel is the cause, and their appearance in the fp16 log is a
# correlation. The fp8 GEMM dispatch is also identical in every arm -- 60
# distinct (path,n,k,m) branches, all fp8_gemm_w8a16/oneDNN -- so the win is not
# in GEMM kernel selection either.
#
# Two things the usual explanations would need, neither of which holds:
#   * "fp16 halves memory traffic" cannot be it. The weights stay fp8 and lm_head
#     stays bf16 in both arms, so the fp16 arm moves no fewer weight bytes per
#     step, and decode here is bandwidth-bound on exactly those bytes. fp16 and
#     bfloat16 are both two bytes anyway.
#   * apply_block_scaled_mm (kernels/linear/scaled_mm/xpu.py) does
#     `a16 = A if A.dtype is float16 else A.to(torch.float16).contiguous()`, so a
#     bf16 arm pays one extra elementwise copy per block-fp8 GEMM call. Measured
#     in the deployed image at 17 us per call at decode shapes (m=1 and m=8,
#     k=5120; launch-dominated, so less once captured in a graph). Whether that
#     explains the delta depends on how many such calls a step makes, which was NOT
#     counted: the census reports 60 distinct (path,n,k,m) shapes, which is not a
#     per-step call count, and inside FULL_DECODE_ONLY graphs the launch overhead
#     that dominates the standalone measurement largely disappears. So this is a
#     plausible contributor at an unquantified size, not an accounting.
# So the win is measured and reproducible, and the "because" is OPEN. Do not repeat
# "fp16 enables the ESIMD kernels, which is why it is faster": that sentence was in
# this file, it sounded explanatory, and the ablations above falsified it. A change
# that makes a faster-looking path appear in the log is not automatically caused by
# that path.
#
# What MAMBA_SSM_CACHE_DTYPE does is established, and it is not redundant with
# DTYPE. Two layers decide the recurrent-state dtype and the later one wins, so
# reason about them in this order:
#   1. cache_config._mamba_state_dtype lets the temporal state INHERIT conv_state
#      when the setting is "auto". Read alone, that suggests DTYPE=float16 would
#      bring the temporal state along for free.
#   2. Qwen3_5ForConditionalGenerationConfig.verify_and_update_config
#      (model_executor/models/config.py:752-777) then rewrites an "auto"
#      mamba_ssm_cache_dtype to the checkpoint's mamba_ssm_dtype, which for this
#      model is float32. It runs before the caches are built, so step 1 never sees
#      "auto" and inheritance never applies.
# Net effect: with both knobs at auto the recurrent state is float32 whatever
# DTYPE says, and DTYPE=float16 alone leaves conv_state float16 / temporal float32
# -- which is why the GDN core below rejects it. The engine says so directly: the
# fp16 arm logs "Qwen3.5 model specifies mamba_ssm_dtype='float32' in its config,
# but --mamba-ssm-cache-dtype='float16' was passed. Using the user-specified
# value." Confirm any future claim here against THAT line in the engine log rather
# than against step 1 in isolation; an earlier version of this comment quoted the
# inheritance rule and a config probe of it, and described a state the engine never
# reaches.
#
# Its own effect is KV capacity, 384,956 -> 414,646 tokens (+7.7%) alongside
# DTYPE=float16, with throughput neutral on its own. It is also what lets the ESIMD
# GDN core run at all -- that guard needs conv_state AND ssm_state float16 -- which
# is worth knowing even though the core is not, per the ablations, what makes it
# fast. Probe placement matters for any claim like this: the census note must sit
# immediately before the op() call, not at function entry. An entry-site note
# printed confidently from an arm where the guard then rejected the call.
#
# Note what is NOT a lever: the per-tensor ESIMD fp8 GEMV/GEMM path in
# kernels/linear/scaled_mm/xpu.py belongs to XPUW8A16FP8LinearKernel, and this
# checkpoint is block-fp8 (128x128), so it selects XPUFp8BlockScaledMMKernel and
# can never reach that code.
#
# Quality did not regress under either arm, judged against the control on the same
# harness: paired GSM8K 29/30 on both arms and control, 19/20 on a 20-item
# exact-answer battery on both with the SAME single miss (an intrinsic model error,
# not a dtype one), 0/20 non-reproducible greedy outputs, and correctness 15/15
# with the acceptance fingerprint unchanged at C1 26 169 7.500.
DTYPE="${DTYPE:-float16}"
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-float16}"
# --dtype is always passed; --mamba-ssm-cache-dtype is omitted entirely when auto.
# With both defaults now float16 the default argv carries both flags -- the point
# of the rule above is that the default IS the measured-best permutation, so do
# not "restore the unchanged argv" by reverting these defaults; revert only if a
# future measurement shows a different permutation wins.
DTYPE_ARGS=(--dtype "$DTYPE")
if [[ "$MAMBA_SSM_CACHE_DTYPE" != "auto" ]]; then
  DTYPE_ARGS+=(--mamba-ssm-cache-dtype "$MAMBA_SSM_CACHE_DTYPE")
fi

# KV_CACHE_DTYPE=auto omits --kv-cache-dtype. fp8 is the only quantized KV dtype
# that works on XPU: int4_per_token_head hits the 128 KiB local-data-shared-memory
# wall in the attention kernel.
# Assign before use: serve.sh runs under `set -u`, so referencing a bare
# $KV_CACHE_DTYPE in the if-body aborts the script when the caller never set it.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
KV_DTYPE_ARGS=()
if [[ "$KV_CACHE_DTYPE" != "auto" ]]; then
  KV_DTYPE_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

# Space-separated escape hatch for one-off flags, e.g.
#   EXTRA_VLLM_ARGS="--language-model-only --mamba-ssm-cache-dtype float16"
# (`read` returns non-zero on an empty value, hence the `|| true` under set -e.)
# This is where the graph ceiling lives; see "Runtime flags" in
# docs/configuration.md.
read -r -a PASS_ARGS <<<"${EXTRA_VLLM_ARGS:-}" || true

serve() {
  vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    "${DTYPE_ARGS[@]}" \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --block-size 64 \
    --enable-prompt-tokens-details \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    "${KV_ARGS[@]}" \
    --default-chat-template-kwargs "{\"enable_thinking\": true, \"preserve_thinking\": true, \"reasoning_effort\": \"xhigh\"}" \
    "${KV_DTYPE_ARGS[@]}" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    "${PASS_ARGS[@]}"
}

# In eager mode Triton kernels are still JIT-compiled during the first
# inferences (layer_norm / postprocess_mamba / _prepare_dflash_inputs were seen),
# which makes the first one or two real requests slow enough to time out. Run
# vLLM in the background, wait for /health, burn ~64 decode steps through a
# throwaway completion (also exercises the DSpark verify path), then supervise.
# WARMUP=0 skips this; do that only when iterating on a cold-start failure where
# the warmup traffic itself is the thing under suspicion.
warmup() {
  local py='
import json,sys,urllib.request
url = sys.argv[1]; model = sys.argv[2]; mx = sys.argv[3]
body = json.dumps({"model": model, "max_tokens": int(mx), "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content":
                     "Count from 1 to 40, separated by spaces. Do not explain."}]}).encode()
req = urllib.request.Request(url, data=body,
                             headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=600).read()
'
  for n in 1 64; do
    if python3 -c "$py" "http://127.0.0.1:${PORT}/v1/chat/completions" \
        "$SERVED_NAME" "$n" >/dev/null 2>&1; then
      echo "[serve.sh] warmup n=$n done"
    else
      echo "[serve.sh] warmup n=$n failed (continuing)"
    fi
  done
}

if [[ "${WARMUP:-1}" != "1" ]]; then
  serve
  exit $?
fi

serve &
VLLM_PID=$!
trap 'kill -TERM "$VLLM_PID" 2>/dev/null || true' TERM INT

for _ in $(seq 1 720); do
  if python3 -c "import urllib.request,sys; urllib.request.urlopen(sys.argv[1], timeout=5).read()" \
      "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    wait "$VLLM_PID"
    exit $?
  fi
  sleep 5
done

warmup
wait "$VLLM_PID"
