# The DSpark drafter: pairing, the config edit, and why `method=dflash` is a trap

Drafter: `rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70`, pinned at commit
`94f8b9cc16571ae53ce0a34494265a8419962cf3`.
`model.safetensors` sha256
`b6c45ca365d983254c8f7a1dfb9c116f176b4ab0a8710180f810a08e2ad335cc`
(2,718,576,122 bytes) — `scripts/download-models.sh` checks this, because if the
weights differ, every acceptance and throughput number in the README becomes
unattributable.

1.36B draft model, 5 layers, `hidden_size` 5120, `vocab_size` 248320,
`target_layer_ids [4, 16, 28, 40, 52]` of `num_target_layers` 64, warm-started
from the released RadixArk DSpark drafter and fine-tuned on hidden states captured
from the vLLM FP8 serving stack (SpecForge DSpark strategy, XPU eager attention,
440 steps at lr 5e-5).

## Why this drafter with this target

The drafter was fine-tuned on **FP8** hidden states, and it shows: pos-0 acceptance
55-84% on the FP8 target vs 45-61% on an INT4 target. FP8 is the smartest of the
quant matrix on this box and the right partner for this drafter; it also costs KV
headroom (KV 9.97 GiB, max concurrency for the full 262k context 1.46x — INT4 gets
15.37 GiB / 2.25x).

Quote the era with the throughput figures from that comparison (suite 15.67 vs
13.56 tok/s): both were measured **eager, k=4, no graphs**, which is not the
deployed shape. The acceptance ratio is the transferable finding; the absolute
numbers describe the eager k=4 shape, not the one this repo ships. The graphed
configuration's own suite metric is ~40-42 tok/s — see the README table.

The pairing is legitimate across quantizations because the drafter's contract with
the target is *only* those hidden-state taps: `target_layer_ids`,
`num_target_layers`, `hidden_size`, `vocab_size`, and `layer_types` at each tap.
The INT4 text config matches all of them. Weight quantization does not change the
activations the projector consumes, which is why C1 acceptance length is identical
(7.500) on both targets and only absolute throughput differs.

## The required config edit

The published `config.json` declares:

```json
"architectures": ["DSparkDraftModel"],
"auto_map": { "AutoModel": "dspark.DSparkDraftModel" }
```

On `0.26.0-b2` neither resolves. The boot dies at pydantic validation of
`SpeculativeConfig` — `Model architectures ['DSparkDraftModel'] are not supported
for now` — before any model load. `profiles/fp8/model-patches/drafter-config.json`
is the corrected file. It changes **exactly two things**:

```diff
-    "DSparkDraftModel"
+    "Qwen3DSparkModel"
-  "auto_map": {
-    "AutoModel": "dspark.DSparkDraftModel"
-  },
```

`Qwen3DSparkModel` is the registered architecture (`registry.py:620` maps it to
`qwen3_dspark/Qwen3DSparkForCausalLM`). `auto_map` must go too: leaving it makes
transformers try to import the shipped `dspark.py` as custom code instead of using
the registered model.

Everything else in the file must stay byte-identical to upstream — `block_size`,
the confidence/markov head settings, `target_layer_ids`, `rope_parameters`. Any
drift there silently changes acceptance and invalidates the comparison. `setup.sh`
loads both files, diffs them in memory, and dies with the differing keys printed if
they are not exactly what is expected.

`dspark_block_size` is deliberately **absent** from the config. If present it
hard-requires `num_speculative_tokens >= 7`, which would remove the ability to
tune k at all. (For the deployed profile k=7 is what you want anyway — see
`cudagraph-stride-defect.md` — but pinning it in the checkpoint makes a legitimate
experiment impossible and an illegal one invisible.)

## `method=dflash` cannot boot with this drafter on 0.26

Not slow — unbootable. 0.26 auto-prefixes `DFlash` onto a draft architecture that
lacks the prefix (`registry.py:613-616` documents the convention), so
`Qwen3DSparkModel` becomes `DFlashQwen3DSparkModel`, which no registry entry
serves. Startup dies at the same pydantic validation, before any model load.
Rewriting only `architectures` to the registered `DFlashDraftModel` (config copy,
weights symlinked so hyperparameters stay identical) still exits during startup.

So on 0.26 a dflash-vs-dspark A/B would measure the config edit, not a supported
configuration. An older framing — "dflash caps acceptance at ~24%" — is superseded:
with this drafter it is worse than slow, it does not start.

`method=dspark` is also the *better* path independently: dspark is V2-runner-only
and natively performs the "anchor is the first prediction" readout that PR #620
hand-patched into V1 `dflash.py` (`num_query_per_req = k` rather than `k+1`;
`is_sample = is_query`). Stock V1 dflash samples at offsets 1..k, putting every
draft one position off and collapsing acceptance to ~24% for any SpecForge-trained
drafter. **With dspark on 0.26 no image patch is needed** — which is why this repo
ships no Python patches at all.

Because dspark is V2-only, `serve.sh` pins `VLLM_USE_V2_MODEL_RUNNER=1`. Pin it in
a no-spec comparison arm too: `config/vllm.py` forces V2 when `method=dspark` but
leaves a no-spec run of this architecture free to default to V1, so without
pinning, an A/B silently measures V2-vs-V1 as much as spec-vs-none.

## What "spec decode is working" actually requires you to check

`vllm:spec_decode_num_drafts_total` must move. A spec arm that records zero drafts
while the engine is alive is a failure, not a measurement — an early version of
this harness recorded `0 0 n/a` for a dead engine and it looked like a valid
no-drafts result. `scripts/bench/acceptance.sh` treats zero drafts in a probe
window as a hard failure and flags accept_len near 1.0 (drafts flowing but all
rejected) as suspect rather than passing it silently.

Reference windows for the deployed FP8 config, `1 + accepted/drafts`:

```
C1-window   26 169 7.500
NAT-window  256 512 3.000
```
