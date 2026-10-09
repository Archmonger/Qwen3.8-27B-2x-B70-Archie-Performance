# Placeholder — INT4 profile

This directory holds **no runnable config** — only these READMEs, which describe
the shape of the future Qwen3.8-27B INT4 profile — so that nothing unverified can
be booted from this repo.

`../../setup.sh --profile int4` refuses with an explanation instead of starting a
container.

## What will go here

```
int4.env                    image, model paths, quantization, k, graph ceiling,
                            KV dtype, oneCCL block
docker-compose.yml          mirrors ../fp8/docker-compose.yml
model-patches/
  drafter-config.json       expected identical to the FP8 profile's: the drafter's
                            contract with the target is hidden-state taps, which the
                            INT4 text config matches exactly
```

The image tag will be pinned separately (`vllm-llmscaler-int4:icr26.35`) if an
INT4-specific image is ever needed, so an FP8 and an INT4 container can never
silently share a tag.

## Known constraints the implementation must respect

1. **Select the checkpoint by what the engine can load, not by the directory
   name.** `quant_method` is the contract, and it has been observed to disagree
   with the name. `auto-round` has no entry in `0.26`'s quantization registry
   (`get_quantization_config("auto_round")` raises `ValueError: Invalid
   quantization method: auto-round`); `gptq`, `gptq_marlin`, and `auto_gptq` all
   resolve. Verify by running each `config.json` through vLLM's own resolver —
   `get_config` + `get_quantization_config` + `from_config` — rather than by
   reading the file.
2. **`method=dflash` does not boot with the published drafter on 0.26.** Use
   `method=dspark`. See [../../docs/dspark-drafter.md](../../docs/dspark-drafter.md).
3. **Pin `VLLM_USE_V2_MODEL_RUNNER=1` in every arm, including a no-spec
   comparison.** Otherwise the A/B measures V2-vs-V1 as much as spec-vs-none.
4. **INT4 buys KV headroom, not necessarily speed.** ~19.7 GiB KV vs FP8's 9.97,
   but the drafter was fine-tuned on FP8 hidden states, so acceptance drops.
5. **Re-derive the graph ceiling** — `MAX_NUM_SEQS x (1 + NUM_SPEC_TOKENS)` —
   and **re-soak**. The oneCCL IPC-handle wedge is a TP2 property of the platform,
   not of the quantization, and `CCL_SYCL_ALLREDUCE_TMP_BUF=1` must stay set.
6. **Correctness is not optional.** The graph-stride defect is invisible to
   throughput and acceptance metrics. Any INT4 run with graphs on must pass
   `../../scripts/bench/correctness.sh`, and until the stride behaviour is
   characterised for the INT4 target, treat `k=7` + `FULL_DECODE_ONLY` + no
   schedule as the only known-safe shape.

See the "Qwen3.8-27B-INT4 Recorded Performance Metrics" and
"Qwen3.8-27B-INT4 Configuration (placeholder)" sections of the README.
