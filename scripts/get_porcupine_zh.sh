#!/usr/bin/env bash
#
# Fetch what Porcupine needs to run a Mandarin wake word.
#
# Porcupine is the better engine for this job — purpose-built, ~0.6% of one
# core, and far less willing than a general recogniser to fire on background
# speech. It is not the default only because it needs a Picovoice account.
#
# This script gets the one piece that is freely downloadable: the Mandarin
# parameter file. Two things it cannot get for you:
#
#   1. An access key.  Free, ~2 minutes: https://console.picovoice.ai
#      Then:  export PICOVOICE_ACCESS_KEY=...
#
#   2. The keyword file for 小艾同学.  Porcupine ships only four built-in
#      Chinese keywords (你好, 咖啡, 水饺, 豪猪), so anything else is generated
#      in the Console: pick language "Chinese", platform "Raspberry Pi", type
#      the phrase, download the .ppn, and drop it at
#          models/xiaoai_zh_raspberry-pi.ppn
#      (or point WakeConfig.porcupine_keyword somewhere else).
#
# Then set backend = "porcupine" in aia/core/config.py.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"

PARAMS="$MODELS/porcupine_params_zh.pv"
if [[ -s "$PARAMS" ]]; then
  echo "already have $(basename "$PARAMS")"
else
  echo "downloading Mandarin parameter file ..."
  curl -fL --progress-bar -o "$PARAMS" \
    "https://raw.githubusercontent.com/Picovoice/porcupine/master/lib/common/porcupine_params_zh.pv"
  echo "installed $PARAMS"
fi

python3 -c "import pvporcupine" 2>/dev/null || {
  echo
  echo "pvporcupine is not installed. In the venv:  pip install pvporcupine"
}

echo
[[ -n "${PICOVOICE_ACCESS_KEY:-}" ]] \
  && echo "PICOVOICE_ACCESS_KEY is set." \
  || echo "PICOVOICE_ACCESS_KEY is NOT set — see the notes at the top of this script."

KEYWORD="$MODELS/xiaoai_zh_raspberry-pi.ppn"
[[ -s "$KEYWORD" ]] \
  && echo "keyword file present: $(basename "$KEYWORD")" \
  || echo "keyword file missing: generate it in the Picovoice Console (see above)."
