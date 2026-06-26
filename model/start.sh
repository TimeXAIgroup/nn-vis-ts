#!/usr/bin/env bash
# Full AudioMNIST 1D-CNN pipeline with sensible defaults.
#   1. download data (idempotent — skips if already present)
#   2. preprocess wavs -> data.npz
#   3. train the 1D CNN
#   4. export TF.js model + layout.json + samples.json into ../js/nn/
#
# Run inside the conda env:
#   /opt/miniconda3/bin/conda env create -f environment.yml   # first time only
#   conda activate audiomnist-nnvis
#   ./start.sh
#
# Override training length with EPOCHS, e.g.  EPOCHS=40 ./start.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

EPOCHS="${EPOCHS:-25}"
export EPOCHS

echo "==> [1/4] download_data.sh"
bash download_data.sh

echo "==> [2/4] preprocess.py"
python preprocess.py

echo "==> [3/4] train.py (EPOCHS=$EPOCHS)"
python train.py

echo "==> [4/4] export_web.py"
python export_web.py

echo "Pipeline complete. Serve the repo root and open cnn/3d.html."
