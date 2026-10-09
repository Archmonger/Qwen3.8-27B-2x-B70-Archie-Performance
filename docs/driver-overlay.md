# Why the image is rebuilt from the pinned base

The pinned base, `intel/llm-scaler-vllm:0.26.0-b2`, is built on
`intel/omix:0.1.0`, which pins the Intel GPU *userspace* driver stack:

```
libze_intel_gpu   1.15.37833   (ICR 26.14)
level-zero loader 1.28.2
IGC               2.32.7
```

The host this runs on carries the Arch `intel-compute-runtime 26.35.39758.10`
userspace pair with a CachyOS `7.2.7` kernel `xe` driver, i.e.

```
libze_intel_gpu   1.17.39758
level-zero loader 1.32.0
IGC               2.41.0
```

**That mismatch does not fail at init.** Container startup succeeds, `/health`
returns 200, and the failure appears only once real kernels run:

- `UR_RESULT_ERROR_DEVICE_LOST` during Inductor/Triton autotune, and
- `UR_RESULT_ERROR_OUT_OF_RESOURCES` inside the TP2 allreduce.

Both read like a VRAM problem or a bad oneCCL setting. Neither is. This cost
several boots to identify because the healthy-looking startup moves the evidence
far from the cause.

## The fix

`image/Dockerfile` copies the userspace driver set out of `intel/omix:0.4.0-devel-ubuntu24.04`
and relinks the SONAMEs:

```
libze_intel_gpu     1.17.39395
libze_loader        1.32.0
libigc / libigdfcl  2.40.13+0
```

Same `1.17` ABI series and same loader minor as the host kernel. Two details that
matter and are easy to get wrong:

- **It must come from an Ubuntu-built image.** The base is Ubuntu 24.04 (noble).
  The host's *Arch*-built `libigc` needs `GLIBCXX_3.4.35`, which noble does not
  have, so copying the host's `/usr/lib` in is the wrong fix even though the
  versions then match perfectly.
- **The build must finish with a load check**, which `image/Dockerfile` does:
  `python -c "import ctypes; ctypes.CDLL('libze_intel_gpu.so.1')"`. If the
  overlay fails, the build fails, instead of a container that boots and then
  faults during autotune.

## What this does NOT need

XPU serving loads **nothing** from `/opt/intel/oneapi` at runtime — the venv
bundles what it needs, and `serve.sh` sources `setvars.sh` only for path setup.
You do not need the oneAPI toolkit installed on the host. Do be aware, though,
that removing the host toolkit (`pacman -Rns intel-oneapi-toolkit`) also removes
`level-zero-loader` and kills GPU init, so if you ever "clean up" the host,
re-check `level-zero-loader` before concluding the container is broken.

## When to rebuild

Rebuild after **either side changes**: a host kernel or `intel-compute-runtime`
bump, or a new `intel/llm-scaler-vllm` base tag. `setup.sh` builds on demand and
verifies the driver loads inside the image; force it with `--rebuild`.

```bash
docker build -t vllm-llmscaler:icr26.35 image/
```

Verify the pinned bases did not move under you (both are pinned by digest on the
measured host):

```bash
docker image inspect intel/llm-scaler-vllm:0.26.0-b2 --format '{{.Id}}'
# sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b
docker image inspect intel/omix:0.4.0-devel-ubuntu24.04 --format '{{.Id}}'
# sha256:ee48e296161b7da989ffbfbbb3390b428e842590eda6258e1f86def3970ef0b3
```

The resulting image ID for the measured build was
`sha256:a267020e746b8eff43c3e0512f135357ea0f1fedd8d70e92b635c913a0417bfa`
(18.3 GB). A rebuild on a different day can differ in layers that do not affect
the engine, so treat the ID as a fingerprint to report, not a gate to enforce.
