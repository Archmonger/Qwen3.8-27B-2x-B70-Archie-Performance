# Verification discipline (and the harness bugs that taught it)

The rule this repo operates on: **acceptance is functional, not status.** An
engine is "working" when it produces the right artifact — correct text, drafts
flowing, a tool call parsed — not when it starts, not when `/health` returns 200,
and not when a config echoes the number you set. This image has no `HEALTHCHECK`
and runs with `network_mode: host`, so `/health` is a liveness signal only.

Several of the traps below are recorded because a broken check cost days, and a
check that cannot fail is worse than no check: it produces confidence.

## What `scripts/verify.sh` proves, and why each check exists

| check | what a pass means | what it catches |
|---|---|---|
| `/health` + container running | something is listening | a crash loop |
| fatal-signature grep of the log | no `Traceback`/`DEVICE_LOST`/`OUT_OF_RESOURCES` | the driver-overlay failure that otherwise surfaces as a "slow boot" |
| engine-printed config | the **engine** has the ceiling / mode / k / method | an env var vLLM silently ignored, or an image rebuild that dropped a flag |
| capture **ladder** contains `MAX_NUM_SEQS x (1+k)` | the saturated batch is actually graphed | the 72-vs-80 defect: echoing `max_cudagraph_capture_size: 80` is not proof the ladder contains 80 |
| `Graph capturing finished ... took N GiB` present | graphs were captured at all | graphs silently off, everything eager |
| V2 model runner confirmed | dspark is actually wired | V1 fallback, where dspark does not exist |
| 15 correctness probes | the text is right | **the stride defect — invisible to every other check** |
| acceptance windows | drafts are flowing and being accepted | spec decode silently disabled |
| 5 serving-surface cells | thinking, tool calls, reasoning, usage details, code | a flag combination that breaks the chat template |
| dtype engagement: exactly one `dtype=torch.X` in the banner, equal to the profile's `DTYPE` | the engine really came up at the shipped dtype | a compose interpolation that dropped the variable, an old `serve.sh` mounted, or an image that reverted the flag. Since the profile now **ships** `DTYPE=float16`, a silent failure here means the box runs the older baseline while every published number claims the faster one |
| ssm-cache engagement (rc 2 printed, never counted) | `--mamba-ssm-cache-dtype` reached the engine, proven by the config warning it emits when it disagrees with the checkpoint | the same drop for the recurrent-state dtype. A pin equal to the checkpoint's own `float32` is **unprovable** and returns 2, so an unprovable claim cannot be read as a verified one |
| host KV tier: `kv_offloading_size` equals the **resolved** `KV_OFFLOAD_SIZE` (committed profile plus the gitignored `host.env` overlay, read through the same `load_host_overlay` `setup.sh` uses), checked in **both** directions, where `0` is an intent rather than a blank | the engine really has the tier at the size this box asked for — or really does not have it at all | the create-time trap in [configuration.md](configuration.md): the value is baked into `Config.Env` at container **create**, so an edited profile or overlay plus a `docker restart` leaves the engine on the old tier with everything else green. Sharing one loader with `setup.sh` is the point: with two copies of the rule, `verify.sh` could demand 0 from a container `setup.sh` created at 50 — a false FAIL — or, the other way round, agree with a boot it never read the same way and hand back a false PASS. vLLM omits the key entirely when the flag was never passed, so the off direction can only be proven by **absence**, which is the one shape a check written as `if [[ -n "$got" ]]` reports as clean. A reported `0.0` is a deliberate FAIL against a wanted `0`: the flag was passed, and `config/vllm.py` builds the connector for anything non-`None`, so it is a zero-byte store rather than tier-off |

## What `setup.sh` refuses to boot

`setup.sh` preflights before it touches the container, and every refusal names the
check that failed rather than exiting 1 — several of these guards exist because the
failure they now catch *looked like something else*. `--dry-run` runs the preflight
and the compose render and changes nothing.

| Refuses to boot when | why that check exists |
|---|---|
| profile is `int4` | exits 3 with an explanation before any build or start: `profiles/int4/` holds no runnable config, so nothing unverified can come up from this repo |
| the profile env file is missing, or the host env overlay is invalid | the overlay may set **only** `KV_OFFLOAD_SIZE`, only numerically — any other key or a non-numeric value exits 2 — and a file named by `KV_HOST_ENV` that does not exist is a hard failure rather than a silent fall back to the committed `0` |
| `python3` is not on `PATH` | the drafter-config and graph-ceiling gates are Python; without it both would pass by not running |
| `B70_BDFS` does not list exactly two well-formed PCI BDFs | one GPU serving TP2 presents as a slow boot, not as a misconfiguration |
| either BDF is not bound to `xe`, or its `by-path` symlink is missing | otherwise it resurfaces later as a "broken driver overlay". The check reads the `driver` symlink with `readlink`, never `cat`: `driver` is a symlink to a *directory*, so `cat` exits non-zero and reports a healthy card as `none` |
| both BDFs resolve to the same card | two workers on one GPU — an OOM far from its cause, blamed on rank 1 |
| the target model dir or a required file is missing, or fewer than 60 safetensors shards are present | an interrupted download otherwise boots a model that cannot load |
| the drafter dir or its `model.safetensors` is missing | same failure one stage later |
| the drafter `config.json` is the unpatched upstream variant | the published drafter ships `architectures: DSparkDraftModel` plus `auto_map`, which on 0.26 resolves to no registry entry — startup dies with `Model architectures ['DSparkDraftModel'] are not supported for now`. `setup.sh` applies `profiles/fp8/model-patches/drafter-config.json` (`Qwen3DSparkModel`, `auto_map` removed) and refuses to start while it is still upstream, so the failure is named here instead of surfacing during weight load |
| the drafter `config.json` differs from the expected patched config in any field | a field-by-field diff, so a near-miss edit that changes the draft width cannot pass a "looks patched" heuristic |
| `max_cudagraph_capture_size` does not equal `MAX_NUM_SEQS x (1 + NUM_SPEC_TOKENS)` — including a value set inside `EXTRA_VLLM_ARGS` | the 72-vs-80 defect: a ceiling below the saturated batch runs the saturated batch eager, and the only symptom is a tok/s shortfall |
| `SPEC_TOKENS_SCHEDULE` is set | with graphs on it garbles output (6–13/30 correct) while the acceptance counters stay healthy — the check that should have caught it did not |
| the compose file does not render, renders an empty device node, or fails to mount the generated `by-path` dir | an interpolation failure is silent at `up` time |
| the image is absent and `--no-build` was given | says which command to run instead of failing to start |
| the in-image level-zero driver cannot load `libze_intel_gpu.so.1` | the Dockerfile's driver overlay did not apply, or the base/driver tags moved. Probed with the venv activated on purpose: an un-activated probe dies with "Failed to infer device type", which is indistinguishable from a genuinely broken driver |
| the image lacks the ported [vllm#52923](https://github.com/vllm-project/vllm/pull/52923) hotfix **and** the host KV tier is enabled | fatal, and unless `--no-build` was given it rebuilds the image itself before proceeding. See [kv-offload-defect.md](kv-offload-defect.md) |
| the container is in a bad state, or no `/health` on `PORT` within the boot timeout | `up -d` returns at create, roughly 200 s before the weights are loaded |
| `scripts/verify.sh` fails | the deployment is not treated as up until post-boot attestation passes |

Severity on the hotfix gate follows the **config**, because that is what decides
whether the defect is reachable: with the resolved `KV_OFFLOAD_SIZE=0` the offloading
connector is never constructed, so an unpatched image cannot fire the assert and only
warrants a warning; with the tier on it is fatal, since the crash takes the whole
engine down and only at long context. A probe that cannot run is classified
`UNKNOWN` and treated as `UNPATCHED` — unable to prove the fix is present is not the
same as proving it is. The gate probes the **image's own bytes**, never the
`Dockerfile` on disk, because the build is skipped whenever the tag already exists
and `./setup.sh` would otherwise happily reuse a pre-hotfix image.

Three conditions **warn** rather than stop, deliberately:

- `SPEC_METHOD` other than `dspark` — `dflash` cannot boot with this drafter, but the
  operator may be pointing at a different model dir.
- An unpatched drafter config under `--dry-run` — a dry run must not be made to fail
  on a state it is about to fix.
- A container **created** with a non-zero host KV tier that is about to be recreated
  at `0`. This is the `host.env`-vanished tripwire: the file is gitignored, so
  `git clean -fdx` deletes it, `setup.sh` and `verify.sh` would both read the same
  missing file and agree with each other, and the box would drop the tier with every
  check green. It cannot be a hard stop — removing the overlay on purpose, to turn
  the tier off, produces an identical signature, and only the operator knows which
  one this is. It is said at the one moment the old value still exists to compare
  against, and names the file to recreate.

## Prove your tests can fail

`scripts/falsify/run-all.sh` drives every guard with inputs that must make it fail:
a synthetic engine log with each attested field corrupted one at a time, a mutated
profile for each `setup.sh` preflight guard, and scripted `/metrics` counters for the
two spec-decode guards. Each harness also runs a **control** — the unmutated input
must pass — because a case that only "discriminates" against an already-broken
baseline proves nothing.

Two real failures this class caught in this project:

- a summary that printed `observed CORRUPT` from **zero** parsed rows (vacuous
  pass — an aggregate over empty input looks like a verdict);
- a boot counter that counted **lines** (17/boot) instead of boots.

And one caught while building this repo: the falsifier's own fixture was written in
a prettified config format, which silently failed every extraction regex — so the
attestation would have reported "all rules bad" on a healthy engine and nobody would
have noticed the regexes matched nothing in production. The fixture is now
transcribed from a real engine log, and `CONTAINER=<name> scripts/falsify/run-all.sh`
cross-checks it against one.

Related shapes to distrust: an unreachable pass condition, an aggregate hash over
empty input, a `grep -qF && good++ || [[ ... ]] && echo WRONG` that prints `WRONG`
on **success** (it parses as `((A&&B)||C)&&D` — use explicit if/else),
`awk 'BEGIN{exit !(x<1.5)}' || fire`, which fires on a healthy value and lets a
broken one through, and `cmd | grep -q .` under `pipefail`, where grep's early exit
SIGPIPEs `cmd` and the pipeline reports failure for content that exists.
`acceptance.sh` tests the predicate positively and says why.

## Harness bugs that misreported a *healthy* engine

All three are fixed with **guards**, not just patches, so the class of bug stays closed:

1. **Correctness reported 12/18 on a good engine.** The prompts were piped through
   `while IFS= read -r`, which split the one multi-line prompt
   (`def add(a, b):\n    return`) into two prompts — so it tested something the
   measured arms never tested. Now: a bash array, plus a hard gate that the run
   executed exactly `${#prompts[@]} * REPS` probes. A run of 12/18 is a harness
   bug and exits non-zero rather than printing a rate.
2. **A soak recorded a healthy engine as TIMEOUT — and, once fixed, nearly recorded a
   wedge as PASS.** `rc=143` was tallied as a stall after the script's own kill. Now:
   `rc=143` is INCONCLUSIVE, and a run with any inconclusive cycle exits 2 rather than
   reporting PASS. (`timeout --preserve-status` is deliberately *not* used — it would
   report 143 on a genuine timeout and destroy the 124-vs-143 distinction.)
   Two further defects in the same script were found while re-recording the soak row:
   it referenced an unset `$MNAME` under `set -u` and aborted cycle 1 (a clean exit, so
   a run that applied *no load at all* read as an engine problem); and it applied one
   burst per cycle, which used to fill the window but drains in ~75 s now that CUDA
   graphs doubled decode — so most cycles came back INCONCLUSIVE, i.e. the soak spent
   87% of its time idle while claiming to soak.
   The fix repeats the load until the deadline, which exposed a *third* bug worth
   naming: reaching the deadline is not evidence of a healthy engine. An engine wedged
   inside oneCCL also reaches 600 s, keeps answering `/health`, and completes nothing —
   so the naive pass condition was satisfiable by the very failure the soak exists to
   catch. A cycle now requires **both** the elapsed window **and** at least one load
   round that completed under its own power; zero completed rounds is a FAIL, not a
   pass. `--load-cells`, `--cycles` and `--cycle-seconds` are overridable so the guard
   can actually be exercised rather than assumed.
3. **A `--max-tokens` default silently changed the metric, and the engine was blamed.**
   `conc.py` computed `aggregate_tok_s` as counted `completion_tokens / wall`, but
   wall time also pays for prefill. Its default was 96 while every recorded number —
   and both scripted callers — used 256, so a hand-run sweep compared 96 against a
   256 record and came out ~13% low. Because the fixed prompt is a few hundred tokens
   (server-reported `prompt_tokens`: 286 counting, 296 planning and prose, 347 coding),
   the uncounted prefill cost amortises over `max_tokens`, so the loss appeared at
   `conc=10` and barely at `conc=1` — which is exactly the signature of a "host-side
   condition". (This section used to cite a "~243-token" prompt and attribute the
   single-stream immunity to prefix caching. Both were wrong: 243 is the `wc -w` *word*
   count of the planning file, and prefix caching is not engaged at these sizes at all,
   so it cannot be making prefill free. The measured ~13%/~4% asymmetry stands; the
   mechanism does not. See [measurement-notes.md](measurement-notes.md).) Verified on one boot
   with alternating arm order, changing only the flag: conc=10 **193.23-202.94** at
   96 vs **229.16-231.64** at 256, against a recorded 227.13. An out-of-sample point
   at 160 interpolated to ~211 and measured 214.82.
   Now: `CANONICAL_MAX_TOKENS = 256` is the default, `max_tokens` is printed with
   every cell and in `--json`, and a non-canonical value exits 3 unless
   `--confirm-noncanonical` is passed. Proven able to fail both ways by
   `scripts/falsify/conc-cases.sh`, including a mutation check that the harness goes
   red if the guard is deleted. The same class of error also inflated the natural
   probe gap (50.21 quoted at `--max-tokens 512` against a ~59 record taken at 96).

## Measurement rules

- **Measure at the `MAX_NUM_SEQS` cap.** `MAX_NUM_SEQS=10` means a saturated engine
  is always at the 10-request cap; measuring at conc=8 hid a 38% batched loss
  across two deploys.
- **Pair your A/Bs and alternate arm order.** ~5% drift per 10 boots on the natural
  probe. Two claims died once order was controlled.
- **Pin and echo every harness parameter that moves the number.** `--max-tokens`
  changes `aggregate_tok_s` by ~15% (rule 3 above), so a comparison where the two
  sides used different values is not a comparison. This is a third category of
  error, distinct from noise and from drift: the measurement was *precise*, it
  measured a different metric. Before attributing a gap to spread, confirm both
  sides used the same `max_tokens`, the same cells, and the same probe.
- **Measure the noise floor instead of picking a threshold.** Boot the identical
  config as a calibration arm and use its spread. A hardcoded 3% once promoted a
  +3.4% artifact to "falsified" on a probe whose real spread was much larger.
  Compute the floor **per probe**: pooling the counting probe with the natural
  prompt gave a 90% floor that muted a real +38%. Note that any historical spread
  figure predating the `max_tokens` finding is suspect, since a mismatched
  `max_tokens` produces a large, uniform, perfectly reproducible pseudo-spread.
  Reps **on one boot** with every harness canonical spanned ~1% (C1 151.12-151.98,
  natural 58.69-58.92, conc=10 at 256 tokens 229.16-231.64); that is a within-boot
  figure and is deliberately not offered as a session-to-session floor, because it
  holds boot, box and cache state fixed while this host drifts ~5% per 10 boots.
- **State the metric with every number.** `tok_s_1_100_after_ttft` from
  `scripts/bench/suite.py`, `aggregate_tok_s` from `conc.py`, and
  `median_tok_s_after_ttft` from `probe.py` are three different things. Engine-log
  "Avg generation throughput" is an aggregate and comparable to none of them.
- **Count tokens the same way the references do.** Ask for `return_token_ids` so one
  SSE event == one generated token; counting raw SSE *text* events understates a
  spec-decoding server catastrophically (7.6 vs 38.1 tok/s on the same stream).
- **Reject a spec arm with zero drafts while the engine is alive**, and gate liveness
  immediately before each probe. A dead engine recording `0 0 n/a` looks like a
  valid measurement of spec decode with no drafts.

## Reproducing a number from the README

```bash
sudo ./setup.sh --profile fp8                 # attests + verifies on the way up
./scripts/bench/run-all.sh --profile fp8      # idle engine, prints all three metrics
./scripts/soak.sh --cycles 6 --cycle-seconds 600
```

If a check you added cannot be made to fail, it is not a check.
