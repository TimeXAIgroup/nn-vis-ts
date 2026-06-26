#!/usr/bin/env bash
# Idempotent AudioMNIST (Becker et al.) downloader.
# Checks for local data first; only clones the ~1GB repo if it is missing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data/AudioMNIST"
WAV_DIR="$DATA_DIR/data"

# Consider data "present" if there are wav files under data/AudioMNIST/data/.
if [ -d "$WAV_DIR" ] && [ -n "$(find "$WAV_DIR" -name '*.wav' -print -quit 2>/dev/null)" ]; then
    n=$(find "$WAV_DIR" -name '*.wav' | wc -l | tr -d ' ')
    echo "AudioMNIST data already present ($n wavs at $WAV_DIR) — skipping download."
    exit 0
fi

echo "AudioMNIST data not found locally."
echo "Cloning https://github.com/soerenab/AudioMNIST (~1GB, 30000 wavs)..."
echo "On a slow connection this takes a while; re-running this script is safe."
mkdir -p "$SCRIPT_DIR/data"
git clone --depth 1 https://github.com/soerenab/AudioMNIST "$DATA_DIR"
echo "Done. Wavs are in $WAV_DIR"
