# Dispatch census modules

These four files are **modified copies of vLLM modules**, bind-mounted over the
image's own copies by `scripts/bench/arm.sh` when an arm runs with
`PROBES=instrument`. They exist so an A/B can prove a code path actually executed
instead of inferring it from throughput — a knob that silently did nothing and a
knob that did nothing useful both read "no effect", and only one of them is true
(see `../../docs/measurement-notes.md`).

They are not imported by anything in this repo at runtime. Nothing here is on the
serving path.

## Provenance

Copied from `vllm-llmscaler:icr26.35` (== `intel/llm-scaler-vllm:0.26.0-b2`),
vLLM `0.26.1.dev0+g568afb3a1.d20260907`. Same Apache-2.0 licence as the
upstream; the original `SPDX-License-Identifier` headers are retained.

To regenerate the pristine copies and see exactly what diverged:

```bash
V=/opt/venv/lib/python3.12/site-packages/vllm
docker run --rm --entrypoint /bin/bash -v "$PWD/pristine:/out" vllm-llmscaler:icr26.35 -lc "
  set -e
  cp $V/model_executor/layers/layernorm.py            /out/layernorm.py
  cp $V/v1/attention/backends/flash_attn.py            /out/flash_attn.py
  cp $V/model_executor/layers/mamba/gdn/_xpu_esimd.py  /out/_xpu_esimd.py
  cp $V/model_executor/kernels/linear/scaled_mm/xpu.py /out/xpu.py
"
for f in layernorm.py flash_attn.py _xpu_esimd.py xpu.py; do
  diff -u "pristine/$f" "$f"
done
```

Pristine digests, so a mismatch above is attributable rather than mysterious:

```
a6db322ccfb567e54fefb2fccb1ce7e6410384d91a78216eb58f592c798ea3b1  layernorm.py
7c9a1b31c4a135a62326346836905db020999b390315dd1d3a168ff3dc43dc50  flash_attn.py
12358c42bb06fc2226e8dbe48ba7adbf31a62be1b52ed80b831a4e2722480e30  _xpu_esimd.py
81f509f1f857cd3d9101b6a76d6a1e157c1910b31880185dda8fa32042c35541  xpu.py
```

## What each file changes

| file | lines removed | effect |
|---|---|---|
| `layernorm.py` | 0 | adds a census note beside the ESIMD fused add-RMS-norm `op()` call |
| `flash_attn.py` | 0 | adds a note beside the ESIMD paged-attention launch |
| `_xpu_esimd.py` | 0 | adds notes beside the GDN decode, GDN **speculative**, and GDN fp16 **`in_proj_ba`** `op()` calls |
| `xpu.py` | **3** | census notes **plus** behaviour-relevant edits, see below |

Only `norm_esimd` and `gdn_spec_esimd` have ever actually printed in an arm. The
paged-attention probe in `flash_attn.py` has never fired on this model, and the
reason is checkable rather than mysterious: its gate accepts only
`gqa_ratio == 2` or `gqa_ratio >= 4 and gqa_ratio % 4 == 0`, while this checkpoint
is 24 query heads over 4 KV heads, i.e. `gqa_ratio = 6`, which satisfies neither
arm. So its silence is a property of the model, not a broken probe and not a dead
kernel. Do not read the absence as an engagement failure — and do not use it as a
`NEEDLE`, which would abort an arm for a reason unrelated to the knob under test.

`arm.sh` refuses to mount a probe directory that has no `SHA256SUMS`, or whose
files do not match it. That is not ceremony: three of these four are pure logging
and harmless, but the fourth is not, and a stale or half-edited probe directory
once produced a confident false claim about which kernel ran.

### `xpu.py` is not purely additive

It replaces three module constants with env-overridable ones, and adds an env
hook to extend the blockscale allowlist:

```python
_XPU_BLOCK_W8A16_MIN_M = int(os.environ.get("VLLM_XPU_BLOCK_W8A16_MIN_M", "128"))
_XPU_BLOCK_ONEDNN_MIN_M = int(os.environ.get("VLLM_XPU_BLOCK_ONEDNN_MIN_M", "8"))
_XPU_BLOCK_QWEN36_27B_ONEDNN_MIN_M = int(
    os.environ.get("VLLM_XPU_BLOCK_QWEN36_27B_ONEDNN_MIN_M", "4")
)
# and VLLM_XPU_BLOCK_EXTRA_SHAPES="5120x17408,..." to add allowlist entries
```

The **defaults are byte-identical to the image's values**, so an arm that sets
none of those variables dispatches exactly as the unpatched image does. That is
the property to check first if these numbers ever stop matching the recorded
table: the whole point is that the probe is inert unless explicitly asked to
change something. If you edit these files, keep that property or say plainly that
you broke it.

## Probe placement

A census note belongs **immediately before the `op()` invocation**, never at
function entry. An earlier revision put the GDN note at the top of the entry
points, before the block that checks `conv_state.dtype`/`ssm_state.dtype`; it
printed confidently from an arm where the guard then rejected the call and the
kernel never ran. Entry proves a function was *called*; only a probe beside the
launch proves it *ran*.

Notes deduplicate per distinct `detail`, so a count of 11 means 11 distinct
row-count buckets — "distinct shapes seen", not invocations. Use them for
engagement, never for call counts.
