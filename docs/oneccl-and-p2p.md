# oneCCL wedges, P2P, and the knobs that look load-bearing but are not

Read this before changing any `CCL_*` variable, and before copying another
replicator's oneCCL block.

## The wedge: an IPC-handle exchange spin

On TP2 the two workers can hang inside oneCCL's `ze_handle_manager` during the
IPC-handle exchange. It is a **spin, not a crash**: the engine stops generating
while `/health` keeps returning 200, and it **does not self-heal** (53/53
observed occurrences had `restart_count=0`, so nothing recovers on its own and a
liveness probe will never catch it). It reproduces with a cold (1-entry) handle
cache, and it is load-shaped — most often seen shortly after boot under
saturation.

### The one knob that works

```
CCL_SYCL_ALLREDUCE_TMP_BUF=1
```

This is the only mitigation found. Evidence: in a same-run paired test, the
allreduce-buffer arm wedged **0/6** times while its control wedged **4/5**
(Fisher p=0.0152); an earlier independent run measured 0/5 vs 5/5. The shipped
config then soaked **6/6 sustained cycles, ~3894 s of continuous conc=16/conc=8
load, 0 wedges, 0 restarts**, matching the mitigation baseline.

`scripts/soak.sh` reproduces that soak. Six clean cycles lower the risk; they do
not eliminate it. Keep the variable set — `profiles/fp8/fp8.env` pins it, and
`serve.sh` defaults it to 1.

Paired with it, and not optional:

```
CCL_SYCL_ALLGATHERV_TMP_BUF=0
CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072
CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576
```

`ALLGATHERV_TMP_BUF=0` is a hard gate on this hardware (Battlemage): pinning the
allgatherv buffer faults. Do not "harmonise" it with the allreduce setting.

## `CCL_NO_P2P=1` — does not work here, and here is why that matters

A public replication of a similar deployment needed a **no-P2P** oneCCL path
because *its* host hung during compilation with `CCL_TOPO_P2P_ACCESS=1` +
`drmfd`; privileged whole-DRI access did not help. That block pins `lo`, which
does exist inside a container, so it was worth testing rather than dismissing on
"host interface names" grounds.

Measured here: **it cannot start at all.** Every variant reaches

```
oneCCL: ze_handle_manager.cpp:42 mem_to_ipc_handle: EXCEPTION:
device_fd is invalid value
```

under `CCL_ZE_IPC_EXCHANGE=pidfd`, under sockets, and with
`ZE_FLAT_DEVICE_HIERARCHY=FLAT` as well as `COMPOSITE`. The identical block on
the original author's own image fails the same way, so it is that container's
device-fd situation, not the base image.

Two conclusions worth keeping:

- The no-P2P requirement came from a PCIe topology this box does not share. It is
  a **hardware-specific workaround, not a port**.
- The upside was already bounded: TP=1 no-spec measured 10.13 tok/s vs TP2 8.60,
  so collectives are cheap here and the whole lever is worth ~1.5 tok/s even
  hypothetically. Before copying someone's comm settings, measure what comm costs
  on *your* topology.

The switch is kept in `serve.sh` purely so the failure can be reproduced. Default
stays P2P: `CCL_TOPO_P2P_ACCESS=1`, `CCL_ZE_IPC_EXCHANGE=pidfd`,
`CCL_ENABLE_SYCL_KERNELS=1`.

## `CCL_ATL_*` / `FI_TCP_IFACE` host interface pinning

Dropped. Inside the container the exchange mode is irrelevant until `by-path` is
filtered (see `dri-by-path.md`) — oneCCL aborts in `fill_device_fds` on dangling
entries before CCL tuning ever matters. `serve.sh` explicitly unsets
`CCL_ATL_TRANSPORT`, `CCL_ATL_SRC_IFACE`, and `FI_TCP_IFACE` on the P2P path so a
host-level export cannot leak in.

## The three XPU env knobs: all inert

`VLLM_XPU_ENABLE_XPU_GRAPH`, `VLLM_XPU_USE_SAMPLER_KERNEL`, and
`ZE_FLAT_DEVICE_HIERARCHY` were each swept against the deployed config. None
improved on baseline; one arm only looked like it had until paired testing
disproved it (boot-order drift, not the knob). They are wired into `serve.sh` as
`XPU_GRAPH` / `XPU_SAMPLER` because they are needed to *enable* graphs at all, not
because tuning them helps.

`XPU_SAMPLER` exists separately from `XPU_GRAPH` for one reason only: the image
bundles the sampler kernel switch with the graph switch for historical reasons,
and separating them lets a graph run be attributed between capture and sampler.

## Boot drift is real and will find you

The box drifts ~5% per 10 boots on the natural probe. Every A/B in these docs is
**paired** with alternating arm order for exactly this reason. Two claims died once
drift was controlled: an "env knob win" and a "graphs are slower" reading. If a
delta is under ~3 tok/s on the natural metric, it is not a result until it survives
paired repeats.

An earlier revision of this file also claimed "a single live engine shows ~14%
peak-to-trough spread". That figure is not trusted and has been dropped: it predates
the discovery that `conc.py`'s `--max-tokens` default silently changed the metric,
and a mismatched `max_tokens` produces exactly this kind of large, uniform,
reproducible pseudo-spread. Reps on one boot with every harness pinned to its
canonical parameter varied by ~1% (natural 58.69-58.92, conc=10 at 256 tokens
229.16-231.64), but that is reps-on-one-boot, not a validated session-to-session
noise floor, so it is not offered as a threshold either. Measure the floor for the
comparison you are actually making, with the identical config as a calibration arm,
and check that both sides used the same `max_tokens` before calling anything noise.
