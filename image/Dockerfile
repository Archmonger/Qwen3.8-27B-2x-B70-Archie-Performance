# intel/llm-scaler-vllm:0.26.0-b2 with the Intel GPU userspace driver stack
# upgraded from the newer OMIX 0.4.0 base (same Ubuntu 24.04, so same libstdc++).
#
# Why: llm-scaler 0.26.0-b2 is built on intel/omix:0.1.0, which pins
#   libze_intel_gpu 1.15.37833 (ICR 26.14) + level-zero loader 1.28.2 + IGC 2.32.7
# while the measured host's kernel xe driver (CachyOS 7.2.7, Arch
# intel-compute-runtime 26.35.39758.10) pairs with
#   libze_intel_gpu 1.17.39758 + loader 1.32.0 + IGC 2.41.0.
# That UMD/KMD mismatch does not fail at init; it faults only once real kernels
# run -- UR_RESULT_ERROR_DEVICE_LOST during Inductor/Triton autotune and
# UR_RESULT_ERROR_OUT_OF_RESOURCES inside the TP2 allreduce.
#
# OMIX 0.4.0 ships libze_intel_gpu 1.17.39395 + loader 1.32.0 + IGC 2.40.13:
# the same 1.17 ABI series and the same loader minor as the host kernel, and
# Ubuntu-built so it links against the base image's libstdc++ (the host's Arch
# build needs GLIBCXX_3.4.35, which noble does not have).
#
# Rebuild after either side changes:  docker build -t vllm-llmscaler:icr26.35 .
#
# Both inputs are pinned by digest, not by tag alone. A tag on a registry is
# mutable, and a silently-moved tag produces the worst failure this file exists to
# prevent: the build succeeds, the container starts, /health answers 200, and it
# then faults with DEVICE_LOST during autotune -- indistinguishable from the
# UMD/KMD mismatch the overlay below fixes. The tags are documentation; the
# digests are what get pulled. Both were read from the measured host with
# `docker images --digests`. To try a different stack, override explicitly:
#   docker build --build-arg BASE=intel/llm-scaler-vllm:0.26.0-b3 -t ... image/
# and re-run scripts/verify.sh -- it reads the engine's own config, so a moved
# base cannot pass as the measured configuration.
ARG BASE=intel/llm-scaler-vllm:0.26.0-b2@sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b
ARG DRIVER_SRC=intel/omix:0.4.0-devel-ubuntu24.04@sha256:ee48e296161b7da989ffbfbbb3390b428e842590eda6258e1f86def3970ef0b3

FROM ${DRIVER_SRC} AS driver-src

FROM ${BASE}

COPY --from=driver-src \
    /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395 \
    /usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0 \
    /usr/lib/x86_64-linux-gnu/libze_tracing_layer.so.1.32.0 \
    /usr/lib/x86_64-linux-gnu/libze_validation_layer.so.1.32.0 \
    /usr/lib/x86_64-linux-gnu/libigc.so.2.40.13+0 \
    /usr/lib/x86_64-linux-gnu/libigdfcl.so.2.40.13+0 \
    /usr/lib/x86_64-linux-gnu/

RUN set -eux; \
    cd /usr/lib/x86_64-linux-gnu; \
    ln -sf libze_intel_gpu.so.1.17.39395    libze_intel_gpu.so.1; \
    ln -sf libze_intel_gpu.so.1             libze_intel_gpu.so; \
    ln -sf libze_loader.so.1.32.0           libze_loader.so.1; \
    ln -sf libze_loader.so.1                libze_loader.so; \
    ln -sf libze_tracing_layer.so.1.32.0    libze_tracing_layer.so.1; \
    ln -sf libze_tracing_layer.so.1         libze_tracing_layer.so; \
    ln -sf libze_validation_layer.so.1.32.0 libze_validation_layer.so.1; \
    ln -sf libze_validation_layer.so.1      libze_validation_layer.so; \
    ln -sf libigc.so.2.40.13+0              libigc.so.2; \
    ln -sf libigc.so.2                      libigc.so; \
    ln -sf libigdfcl.so.2.40.13+0           libigdfcl.so.2; \
    ln -sf libigdfcl.so.2                   libigdfcl.so; \
    rm -f libze_intel_gpu.so.1.15.37833 libze_loader.so.1.28.2 \
          libze_tracing_layer.so.1.28.2 libze_validation_layer.so.1.28.2 \
          libigc.so.2.32.7+0 libigdfcl.so.2.32.7+0; \
    ldconfig; \
    ls -la libze_intel_gpu.so* libze_loader.so* libigc.so*; \
    python -c "import ctypes; ctypes.CDLL('libze_intel_gpu.so.1'); print('driver loads OK')"

# --- KV-offload hotfix: vllm-project/vllm#52923, ported ---------------------
#
# The host KV tier (--kv-offloading-size) kills EngineCore at long context:
# _build_store_jobs slices offload_keys and offload_block_ids from the same
# request state and asserts they are equal, but blocks are allocated for a chunk
# before Request.block_hashes yields that chunk's offload key, so the block slice
# can be longer and the assert fires. Upstream fixed it in #52923 (merged
# 2026-09-03, fixes #52170); this image's vllm commit (568afb3a1, 2026-07-26)
# predates that merge, so the defect is live here. See ../docs/kv-offload-defect.md.
#
# The patch cannot be applied verbatim: this build's storable_chunks() predates the
# PR's *base* too -- it has neither of the two extra bounds upstream applies. So the
# patch is a port (see its header), and these two hashes are what make the port
# trustworthy rather than aspirational:
#
#   PRE  asserts we are patching the exact file the port was written against. If the
#        pinned base moves, this fails HERE, loudly, instead of producing an image
#        with a half-applied patch -- or, worse, `patch` fuzzing a hunk into the
#        wrong place and the build succeeding.
#   POST asserts the result is the exact bytes the port was reviewed as. The ported
#        storable_chunks() is byte-identical to upstream's post-fix function.
#
# Both are compared against the file at the path vllm itself reports, not a
# hard-coded site-packages path, so a relocated venv cannot silently skip the patch.
# The __pycache__ entry is removed because its .pyc is timestamp-validated: leaving a
# stale one next to a fresh .py is how a patched file gets ignored at import.
#
# The last step IMPORTS the patched module rather than merely parsing it. ast.parse
# passes files that cannot load: this port adds an unquoted `group_state:
# RequestGroupState` annotation, and an annotation whose name is not in scope at
# def-time is a NameError at import time -- after PRE matched, after POST matched,
# and after a parse-only check had printed its OK. PYTHONDONTWRITEBYTECODE keeps
# that import from baking the whole vllm chain as .pyc into this layer. Importing
# the module also arity-checks every storable_chunks() call site, which no hash can.
ARG KV_OFFLOAD_PATCH_PRE=89ac26a80fbc29b9bcaa5a0daba88fb6309247f9bb053e48d2d61efa7d9d66f1
ARG KV_OFFLOAD_PATCH_POST=9179a76b383779e8c6dc8d993543eb8fe77cc60c7599dc0436c1b0a0d0e5d904
COPY patches/kv-offload-52923.patch /tmp/kv-offload-52923.patch
RUN set -eux; \
    V="$(PYTHONDONTWRITEBYTECODE=1 python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"; \
    F="$V/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"; \
    test -f "$F"; \
    printf '%s  %s\n' "$KV_OFFLOAD_PATCH_PRE"  "$F" | sha256sum -c -; \
    ( cd "$V" && patch -p1 --no-backup-if-mismatch -i /tmp/kv-offload-52923.patch ); \
    printf '%s  %s\n' "$KV_OFFLOAD_PATCH_POST" "$F" | sha256sum -c -; \
    rm -rf "$V/distributed/kv_transfer/kv_connector/v1/offloading/__pycache__"; \
    rm -f /tmp/kv-offload-52923.patch; \
    python -c "import ast,sys; ast.parse(open(sys.argv[1]).read()); print('kv-offload hotfix parses OK')" "$F"; \
    PYTHONDONTWRITEBYTECODE=1 python -c "import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as m; print('kv-offload hotfix imports OK', m.RequestGroupState.__name__)"

LABEL org.opencontainers.image.title="llm-scaler-vllm + OMIX 0.4.0 GPU driver stack + vllm#52923 KV-offload hotfix"
