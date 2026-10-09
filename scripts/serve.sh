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
KV_OFFLOAD_SIZE="${KV_OFFLOAD_SIZE:-50}"

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
if [[ "${KV_OFFLOAD_SIZE:-50}" != "0" ]]; then
  KV_ARGS+=(--kv-offloading-size "$KV_OFFLOAD_SIZE")
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
    --tensor-parallel-size "$TP_SIZE" --dtype auto \
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
