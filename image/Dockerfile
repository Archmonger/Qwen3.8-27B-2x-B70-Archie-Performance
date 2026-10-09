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

LABEL org.opencontainers.image.title="llm-scaler-vllm + OMIX 0.4.0 GPU driver stack"
