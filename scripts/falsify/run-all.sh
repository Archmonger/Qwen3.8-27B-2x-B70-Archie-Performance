#!/usr/bin/env bash
# Run every falsification harness in this directory.
#
# The rule these exist to enforce: a check that cannot fail is not a check. Two
# verdict paths in this project's history printed confident results from inputs that
# should have scored nothing, so nothing here is allowed to be "probably fine".
#
# Run this before trusting setup.sh or verify.sh, and after editing any guard or
# attestation rule. See docs/verification.md.
#
# Usage:
#   scripts/falsify/run-all.sh                       # synthetic only
#   CONTAINER=vllm-llmscaler scripts/falsify/run-all.sh   # + cross-check vs the real log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fails=0
for h in "$HERE"/verify-cases.sh "$HERE"/setup-guards.sh "$HERE"/acceptance-cases.sh \
         "$HERE"/conc-cases.sh "$HERE"/prefill-cases.sh "$HERE"/arm-cases.sh \
         "$HERE"/engage-cases.sh "$HERE"/kv-patch-cases.sh "$HERE"/host-env-cases.sh; do
  echo
  echo "################ $(basename "$h")"
  # 1800s covers kv-patch-cases' mutant-image build; the synthetic harnesses finish
  # in seconds, so a timeout here almost always means docker is hanging, not slow.
  if timeout 1800 bash "$h"; then
    :
  else
    echo "^^^ $(basename "$h") FAILED"
    fails=$((fails+1))
  fi
done

echo
if [[ $fails -eq 0 ]]; then
  echo "falsify/run-all: PASS -- every guard and attestation rule can fail"
  exit 0
fi
echo "falsify/run-all: FAIL ($fails harness(es) red)"
exit 1
