#!/usr/bin/env python3
"""Extract the single JSON object embedded in a raw command stdout capture.

Why this exists instead of ``grep -o '{.*}'`` or ``json.loads(open(f).read())``:

* ``probe.py`` prints INDENTED json. A single-line brace match like ``{.*}``
  therefore returns the lone character ``{`` and the arm looks unparseable. The
  capture pipeline used to survive this only because ``2>&1 | tr -d '\\n'``
  flattened the pretty-print as an unintended side effect.
* ``json.loads`` of the whole file rejects a capture that contains a diagnostic
  line as well as a complete payload, so a healthy boot gets discarded. These
  files are raw stdout redirects, and the harnesses do print to stdout around the
  payload -- ``conc.py``'s ``WARMUP_FAILED`` line is one.

So every ``{`` is treated as a candidate start and the first one that decodes
wins. A diagnostic may itself contain braces, which is why a single ``find`` is
not enough.

Used from two places, and both must agree: ``scripts/bench/arm.sh`` extracts the
values it records through the CLI, and ``scripts/lib/degenerate.sh`` decides
whether to reject the arm through the importable function. If they diverge, a
measurement is either recorded unchecked or thrown away.

CLI: prints the compact JSON object on stdout, exits 1 when the file holds none.
"""

import json
import sys


def jsonobj(path):
    """Return the first JSON object decodable from the file at *path*, else None."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i >= 0:
        try:
            return decoder.raw_decode(text[i:])[0]
        except ValueError:
            i = text.find("{", i + 1)
    return None


def main(argv):
    if len(argv) != 2:
        print("usage: jsonobj.py <file>", file=sys.stderr)
        return 2
    obj = jsonobj(argv[1])
    if obj is None:
        return 1
    print(json.dumps(obj))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
