# `dri-by-path` — the filtered DRM directory

`setup.sh` generates this per profile at `$RUNSTATE_DIR/dri-by-path/` and the
compose file mounts it at `/dev/dri/by-path`. **Do not mount the host's
`/dev/dri/by-path` instead.** This is the single most expensive gotcha in the
whole stack, and it fails twice in two different ways depending on how you get
there.

## Why

oneCCL's level-zero backend enumerates `/dev/dri/by-path` to map its ranks onto
physical devices. If that directory contains entries for GPUs the container is
not meant to use, the entries dangle (the container has no `renderD*` for them)
and oneCCL dies.

On the measured host, `/dev/dri/by-path` holds five devices:

| BDF | device | driver | wanted? |
|---|---|---|---|
| `0000:03:00.0` | Arc Pro B70 (Battlemage G31) | `xe` | **yes — rank 1** |
| `0000:83:00.0` | Arc Pro B70 (Battlemage G31) | `xe` | **yes — rank 0** |
| `0000:43:00.0` | Arc A310 (DG2) | `i915` | no |
| `0000:c5:00.0` | RTX 2000 Ada | `nvidia` | no |
| `0000:c7:00.0` | ASPEED BMC VGA | `ast` | no |

Two failure modes, both observed:

1. **In a container:** whole-dir mount, oneCCL aborts in `fill_device_fds` on the
   dangling entries — before any CCL tuning matters. Looks like a oneCCL config
   problem; it is a mount problem.
2. **Bare metal:** with the A310 present, an `L0::Device*` vector assert fires.
   The A310's `i915`/`ze` driver exposes no `zes` twin, so level-zero enumerates a
   device it cannot open.

Filtering to exactly the intended GPUs fixes both. Note this is independent of the
`devices:` list: the container needs *both* the device nodes and a matching
`by-path` view.

## What setup.sh does

It resolves each B70 BDF to its `cardN`/`renderDN` via the host's `by-path`
symlinks (never by scanning `/dev/dri`, because render numbering is not stable
across boots or across the set of GPUs present), asserts both are bound to `xe`,
then writes exactly four symlinks:

```
pci-0000:03:00.0-card   -> ../card2
pci-0000:03:00.0-render -> ../renderD129
pci-0000:83:00.0-card   -> ../card1
pci-0000:83:00.0-render -> ../renderD128
```

and refuses to start if a link points at a node that does not exist on the host. It is
regenerated from scratch on every run, because a stale symlink into a device that
moved would mount a foreign GPU and reintroduce failure mode 1.

Worth naming, because it is a trap: these links are **supposed to dangle on the
host**. `../card2` resolves only inside the container, where the directory is mounted
at `/dev/dri/by-path` and `../card2` is the `/dev/dri/card2` that the compose
`devices:` list passes through. An early version of `setup.sh` validated them with
`[[ -e ]]` on the host, which therefore aborted every correct setup with "dangling
by-path symlink". Validate the *target name against `/dev/dri`* — that is the thing
the container actually needs handed to it.

## On your box

`setup.sh` derives everything from `B70_BDFS` (or `--b70-bdfs`), so the only
thing to change is which two BDFs are your B70s:

```bash
lspci -nn | grep -i '8086:e223'          # Battlemage G31 / Arc Pro B70
ls -l /dev/dri/by-path/                  # confirm the -card / -render names
sudo ./setup.sh --b70-bdfs 0000:03:00.0,0000:83:00.0
```

Rank order follows PCI bus id, so the lower BDF is rank 0. If a third `xe` card
or a second Arc SKU shows up in `by-path`, it is excluded automatically — the
filter is by the BDF list, not by "everything that looks like an Intel GPU".
