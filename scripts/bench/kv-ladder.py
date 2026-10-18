#!/usr/bin/env python3
"""Long-context ladder for the host KV tier -- the gate the vllm#52923 hotfix needs.

Why this exists: the offloading connector's key/block desync only fires on LONG
prompts. Short traffic passed scripts/verify.sh, a correctness battery, and hours
of ordinary load before the defect showed up, so nothing in the pre-deploy gate
exercised a 100k+ prompt. This is that missing gate. It is the cell that killed
EngineCore 3/3 before the ported fix, so a PASS here is the only thing that says
`KV_OFFLOAD_SIZE=50` is safe to ship.

What a PASS means, and the three ways this harness could otherwise lie:

  * /health is never evidence. With restart=unless-stopped, a dead-worker engine
    keeps answering 200 (the zombie-engine trap). Acceptance is a real completion
    carrying this rep's planted needle.
  * Every rep gets a unique prefix salt. Prefix caching is ON, so a repeated
    prompt is served from the GPU prefix cache, skips prefill, and therefore
    skips the _build_store_jobs path the assert lives in. An unsalted rep is a
    pass that tested nothing.
  * The server's OWN prompt_tokens must land within --tolerance of the target,
    or the cell is recorded FAIL(vacuous). A hand-estimated chars/token once
    asked for 160k and got 70,901 -- "no crash" on a prompt too short to reach
    the defect is meaningless. The token count is calibrated from one real
    request at start-up for the same reason.

Cells: a single ~160k prompt x3 sequential (3/3 fatal pre-fix), two concurrent
~160k (never measurable pre-fix -- the single-request cell killed the engine
first), and one at 93% of --max-model-len. Container RestartCount and a 24 h
error grep are captured before and after; either moving fails the run.

Usage:
    python3 scripts/bench/kv-ladder.py                     # against the running engine
    python3 scripts/bench/kv-ladder.py --reps 1 --no-docker # quick smoke
Exit 0 = every cell served the right needle at the right token count with the
engine still up. Non-zero = do not ship the tier.
"""
import argparse
import json
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

FILLER = "The quarterly report summarizes regional logistics in the northern district. "
TAIL = "\n\nQuestion: What is the magic number for this audit? Answer with only the number."


def sh(cmd):
    """Run a shell command, returning stdout stripped. Never raises on a miss."""
    return sh_rc(cmd)[0]


def sh_rc(cmd):
    """Run a shell command; return (stdout_stripped, returncode_or_None).

    The pair matters. A helper that collapses "the command found nothing" and
    "the command never ran" into one empty string turns an absent check into
    evidence of absence -- which is exactly how a crash-capture that failed
    because docker was unreachable would print "(no crash lines)" and pass.
    """
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return p.stdout.strip(), p.returncode
    except Exception:
        return "", None


class Engine:
    def __init__(self, base, model, timeout_s):
        self.base = base.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def completions(self, prompt, max_tokens):
        """One non-streamed chat completion. Returns (http_status, json_or_None, error)."""
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            # Thinking is disabled so the needle lands in the visible answer
            # rather than being rationales away before it.
            "chat_template_kwargs": {
                "enable_thinking": False,
                "preserve_thinking": False,
            },
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return r.status, json.loads(r.read().decode()), None
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                return e.code, json.loads(body), None
            except Exception:
                return e.code, None, body[:200]
        except Exception as e:  # timeout, refused, engine gone mid-request
            return 0, None, f"{type(e).__name__}: {e}"[:200]


def draw_needles(n, salt_tag):
    """n distinct 5-digit needles for this run.

    Uniqueness is by construction (a random sample without replacement) rather
    than by an arithmetic formula, because the formula this replaces could collide
    across cells -- and two cells sharing a needle is precisely the case where a
    cached or crossed answer still reads as a pass. Seeded from the run's salt so a
    rerun gets a fresh set: reusing an old needle would let a prefix-cached answer
    satisfy the new cell.
    """
    rng = random.Random(f"kv-ladder-{salt_tag}-{n}")
    return rng.sample(range(10000, 99999), n)


def build_prompt(target_tokens, rep, salt_tag, chars_per_token, needle):
    """A prompt of about target_tokens, with a needle only this cell can answer.

    target_tokens * chars_per_token is the TOTAL budget; the filler goes on both
    sides of the needle, so each half gets half of it.

    The needle is passed in, not derived from (rep, target). The earlier formula
    `4000 + rep*111 + target%100` could hand two cells the same needle -- any
    --max-model-len making int(mml*0.93) % 100 == 0 collided the 93%-of-window cell
    with long_r1 -- and two cells sharing a needle is exactly the case where a
    cached or crossed answer still "passes". main() draws needles without
    replacement, so uniqueness is by construction rather than by luck.
    """
    salt = f"AUDIT-SALT-{salt_tag}-R{rep}-{target_tokens}-UNIQUE-PREFIX-"
    half = max(0, (int(target_tokens * chars_per_token) - len(salt) - len(TAIL) - 64) // 2)
    reps = max(1, half // len(FILLER))
    body = FILLER * reps
    text = f"{salt}{body}The magic number for this audit is {needle}. {body}{TAIL}"
    return text, needle


def answer_has_needle(answer, needle):
    """Match the needle as a whole number, not a substring.

    `str(needle) in answer` lets a short needle pass inside a longer number, so a
    rambling answer containing an unrelated figure could satisfy a cell it was not
    asked about.
    """
    return re.search(rf"(?<!\d){needle}(?!\d)", answer or "") is not None


def run_cell(engine, target_tokens, rep, salt_tag, chars_per_token, tolerance,
             max_tokens, needle):
    prompt, needle = build_prompt(target_tokens, rep, salt_tag, chars_per_token, needle)
    t0 = time.time()
    status, body, err = engine.completions(prompt, max_tokens)
    wall = time.time() - t0
    got = None
    prompt_tokens = None
    if body:
        try:
            got = body["choices"][0]["message"]["content"]
        except Exception:
            got = None
        prompt_tokens = (body.get("usage") or {}).get("prompt_tokens")
    in_range = (
        prompt_tokens is not None
        and abs(prompt_tokens - target_tokens) / float(target_tokens) <= tolerance
    )
    ok = status == 200 and in_range and got is not None and answer_has_needle(got, needle)
    return {
        "rep": rep, "http": status, "wall_s": wall, "prompt_tokens": prompt_tokens,
        "target": target_tokens, "in_range": in_range, "needle": needle,
        "answer": (got or "")[:60], "err": err, "pass": ok,
    }


def report(name, r):
    return (
        f"{name} http={r['http']} wall={r['wall_s']:.0f}s "
        f"prompt_tokens={r['prompt_tokens']} target={r['target']} "
        f"in_range={r['in_range']} needle={r['needle']} "
        f"answer={r['answer']!r} err={r['err'] or '-'} "
        f"=> {'PASS' if r['pass'] else 'FAIL' + ('(vacuous)' if not r['in_range'] else '')}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://localhost:8103")
    ap.add_argument("--model", default="secondary")
    ap.add_argument("--container", default="vllm-arch-fp8")
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--long-tokens", type=int, default=160000)
    ap.add_argument("--reps", type=int, default=3, help="sequential reps of the long cell")
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--max-completion-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--no-docker", action="store_true",
                    help="skip RestartCount/error-grep capture (outside the host)")
    ap.add_argument("--skip-max", action="store_true",
                    help="skip the 93%%-of-window cell")
    args = ap.parse_args()

    engine = Engine(args.base, args.model, args.timeout)
    salt_tag = str(int(time.time()))
    run_t0 = time.time()
    results = []
    fail = []

    def restart_count():
        if args.no_docker:
            return None
        out = sh(f"docker inspect {args.container} --format '{{{{.RestartCount}}}}'")
        return int(out) if out.isdigit() else None

    def err_count():
        if args.no_docker:
            return None
        out = sh(f"docker logs --since 24h {args.container} 2>&1 | "
                 "grep -cE 'AssertionError|UR_RESULT_ERROR_DEVICE_LOST' || true")
        return int(out) if out.isdigit() else None

    if not args.no_docker:
        if not sh("command -v docker"):
            print("[ladder] WARN: no docker CLI; restart/error capture disabled", file=sys.stderr)
            args.no_docker = True

    def mem_avail():
        # Host RAM is part of the record: the tier's cost is host-side, so a run
        # that passed at 80 GiB available and one at 2 GiB are not the same result.
        for line in sh("free -g").splitlines():
            if line.startswith("Mem:"):
                f = line.split()
                if len(f) >= 7:
                    return f"{f[6]}Gib available, {f[2]}Gib used"
        return "mem unknown"

    r0, e0 = restart_count(), err_count()
    print(f"[ladder] baseline restarts={r0} errs24h={e0} {mem_avail()}")

    # Calibrate chars/token from the server's own tokenizer rather than guessing:
    # a hand-estimate once turned a "160k" cell into a vacuous 70,901-token pass.
    chars_per_token = 5.5
    calib_needle = draw_needles(1, salt_tag)[0]
    probe_text, _ = build_prompt(8000, 900, salt_tag, chars_per_token, calib_needle)
    status, body, err = engine.completions(probe_text, 1)
    calib = (body or {}).get("usage", {}).get("prompt_tokens") if body else None
    if calib and calib > 1000:
        chars_per_token = len(probe_text) / calib
        print(f"[ladder] calibrated {calib} tokens / {len(probe_text)} chars "
              f"-> {chars_per_token:.3f} chars/token")
    else:
        print(f"[ladder] WARN: calibration unusable (prompt_tokens={calib}, err={err}); "
              f"keeping {chars_per_token} chars/token -- cells may fail as vacuous",
              file=sys.stderr)

    # One needle per cell, drawn without replacement for the whole run.
    cap = int(args.max_model_len * 0.93)
    cell_names = ([f"long_r{i}" for i in range(1, args.reps + 1)]
                  + ["conc_1", "conc_2"] + ([] if args.skip_max else ["max_window"]))
    needles = dict(zip(cell_names, draw_needles(len(cell_names), salt_tag)))

    print(f"[ladder] R1: single ~{args.long_tokens} x{args.reps} sequential")
    for rep in range(1, args.reps + 1):
        name = f"long_r{rep}"
        r = run_cell(engine, args.long_tokens, rep, salt_tag, chars_per_token,
                     args.tolerance, args.max_completion_tokens, needles[name])
        results.append(r)
        print("[ladder] " + report(name, r))
        if not r["pass"]:
            fail.append(name)

    print(f"[ladder] R2: two concurrent ~{args.long_tokens}")
    conc = {}
    threads = [
        threading.Thread(
            target=lambda k: conc.__setitem__(
                k, run_cell(engine, args.long_tokens, 100 + k, salt_tag,
                            chars_per_token, args.tolerance, args.max_completion_tokens,
                            needles[f"conc_{k}"])),
            args=(k,))
        for k in (1, 2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for k in (1, 2):
        name = f"conc_{k}"
        r = conc[k]
        results.append(r)
        print("[ladder] " + report(name, r))
        if not r["pass"]:
            fail.append(name)

    if not args.skip_max:
        # 93% of the window: the longest prompt the engine will accept.
        name = "max_window"
        print(f"[ladder] R3: {cap} tokens ({cap}/{args.max_model_len})")
        r = run_cell(engine, cap, 1, salt_tag, chars_per_token, args.tolerance,
                     args.max_completion_tokens, needles[name])
        results.append(r)
        print("[ladder] " + report(name, r))
        if not r["pass"]:
            fail.append(name)

    r1, e1 = restart_count(), err_count()
    print(f"[ladder] after restarts={r1} errs24h={e1}")
    # Distinguish "no crash lines" from "could not look". An empty string from a
    # docker that never ran is not evidence that nothing crashed -- and this is
    # the same zombie-engine trap the repo already documents: a dead-worker
    # engine keeps answering /health 200, so the log scan is load-bearing.
    crash_rc = None
    if not args.no_docker:
        # Window comes from the run's own elapsed time, not a fixed 30m: a queued
        # ladder can outlast a hardcoded window and then early crashes age out.
        window_s = max(1800, int(time.time() - run_t0) + 300)
        crash, crash_rc = sh_rc(
            f"docker logs --since {window_s}s {args.container} 2>&1 | "
            "grep -E 'AssertionError|DEVICE_LOST|Traceback' | tail -5")
        if crash_rc is None or crash_rc not in (0, 1):
            print(f"[ladder] CRASH CAPTURE UNAVAILABLE (rc={crash_rc}) -- "
                  "this run cannot attest that the engine did not die")
        else:
            print(f"[ladder] crash-lines (last {window_s}s):\n"
                  + (crash or "  (none)"))
    off = sh(f"curl -s -m 5 {args.base}/metrics | grep kv_offload_total_bytes_total")
    if off:
        print("[ladder] connector counters:\n  " + off.replace("\n", "\n  "))

    # Engine-death attestation is mandatory, not best-effort. If the restart /
    # error capture never ran, a PASS here would be asserting something it never
    # observed -- the exact shape that makes a green harness worse than no
    # harness. --no-docker is an explicit opt-out (off-host smoke runs) and is
    # then reported as an unattested run.
    capture_dead = (not args.no_docker) and (
        r0 is None or r1 is None or e0 is None or e1 is None
        or crash_rc is None or crash_rc not in (0, 1))
    moved = (r1 != r0) or (e1 != e0)
    if capture_dead and not args.no_docker:
        fail.append("engine-death-capture")
    print(f"[ladder] VERDICT cells={len(results)} failed={','.join(fail) or 'none'} "
          f"restart_delta={(r1 - r0) if None not in (r0, r1) else 'n/a'} "
          f"err_delta={(e1 - e0) if None not in (e0, e1) else 'n/a'}")
    if not fail and not moved:
        if args.no_docker:
            print("[ladder] PASS (UNATTESTED): every cell served its needle at the "
                  "right token count, but --no-docker means engine death was not "
                  "observed. Do not treat this as the tier being safe.")
        else:
            print("[ladder] PASS: every cell served its needle at the right token "
                  "count with the engine still up")
        return 0
    print("[ladder] FAIL: do not ship KV_OFFLOAD_SIZE>0 on this image")
    return 1


if __name__ == "__main__":
    sys.exit(main())
