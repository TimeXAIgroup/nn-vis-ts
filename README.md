# nn-vis-ts — interactive 3D visualization of a 1D CNN for spoken digits

A reworking of [Adam Harley's node-link CNN visualization](https://github.com/aharley/nn_vis)
for **time-series / speech**. Instead of drawing an MNIST digit, you pick a spoken digit and watch
the activations of a **1D CNN** trained on [AudioMNIST](https://github.com/soerenab/AudioMNIST)
(Becker et al.) flow through the network in 3D. All inference runs **in the browser** via
TensorFlow.js.

![demo](images/cnn_3d.png)

## Features

- 🎙️ Pick a random AudioMNIST sample (or from a dropdown), **play it**, see its waveform.
- 🧊 Live per-layer **3D activation cubes**, with the input waveform drawn beneath the input row.
- 🔬 Hover a **conv** node → its learned **1D filter** (kernel) plotted as a time series.
- 🌈 Hover an **output** node → **Grad-CAM** highlights the input regions most responsible for that
  class (jet colormap on the last conv layer + projected onto the input + waveform).
- 🏷️ Per-layer **parameter labels** (filters, kernel, stride, output shape) right in the scene.
- 👁️ Toggle any layer's visibility; drag to orbit, scroll to zoom.

## Prerequisites

- [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html) (the commands below assume
  `/opt/miniconda3/`; adjust the path if yours differs).
- `git` and a modern browser with WebGL.
- ~3 GB free disk for the AudioMNIST dataset (only needed to (re)train).

## Setup on a new machine

```bash
git clone <this-repo-url> nn-vis-ts
cd nn-vis-ts

# one-time: create the conda environment (TensorFlow, librosa, etc.)
/opt/miniconda3/bin/conda env create -f model/environment.yml
conda activate audiomnist-nnvis
```

> **Note:** the dataset, trained model and generated web assets are **not** committed (see
> `.gitignore`). You produce them with the pipeline below. (If you prefer a clone to run without
> retraining, remove the `js/nn/...` lines from `.gitignore` and commit those files.)

## Build the model + web assets

```bash
cd model
./start.sh                 # download (idempotent) → preprocess → train → export
# EPOCHS=40 ./start.sh     # train longer for a bit more accuracy
```

`start.sh` runs the full pipeline and writes the three files the web app needs into `js/nn/`:
`tfjs_model/`, `layout.json`, `samples.json`. Individual steps if you want to run them separately:

| Step | Script | Output |
|------|--------|--------|
| Download AudioMNIST (~1 GB git repo, skips if present) | `model/download_data.sh` | `model/data/AudioMNIST/` |
| Resample 48k→8k, fix to 8000 samples, speaker-disjoint split | `model/preprocess.py` | `model/artifacts/data.npz` |
| Train the 1D CNN | `model/train.py` | `model/artifacts/model.keras` |
| Export TF.js model + 3D layout + sample clips | `model/export_web.py` | `js/nn/tfjs_model/`, `layout.json`, `samples.json` |

Notes:
- `export_web.py` converts with the official `tensorflowjs` package if available, otherwise falls
  back to a built-in dependency-free exporter (`model/keras_to_tfjs.py`).
- If no trained model/data exists yet, `export_web.py` still emits an **untrained** model + synthetic
  clips so the UI is renderable (predictions meaningless until you actually train).
- Expected test accuracy with the defaults: **~97%** (held-out speakers).

## Run the demo

Serve the **repo root** over HTTP (the app uses `fetch`, so `file://` won't work):

```bash
# from the repo root
python -m http.server 8000
# then open:  http://localhost:8000/cnn/3d.html
```

### Using it
- **Random sample** / dropdown to choose a clip; press play to hear it.
- **Hover a convolution node** → see that channel's learned 7-tap filter.
- **Hover an output node** → Grad-CAM for that digit class lights up the responsible input regions.
- Use the **Layer visibility** panel to hide/show layers; **drag** to orbit, **scroll** to zoom.

## Architecture

Raw 8 kHz waveform (8000 samples) →
`Conv1D(8, k7, s4)` → MaxPool/4 →
`Conv1D(16, k7)` → MaxPool/4 →
`Conv1D(32, k7)` → MaxPool/4 →
`Conv1D(32, k7)` → MaxPool/4 →
`Flatten` → `Dense(64, ReLU)` → `Dense(10, softmax)`.

- Defined in `model/train.py`; channel counts are kept small so per-layer activations render as 3D
  nodes. Edit it and re-run `start.sh` to change the network — the visualization is **layout-driven**
  (`js/nn/layout.json`) and adapts automatically, including the parameter labels.
- Early layers are long, so each layer's time axis is evenly **subsampled** for display
  (`DISPLAY_LEN_CAP` / `INPUT_DISPLAY_LEN` in `model/config.py`). Hover edges illustrate the
  receptive field; they are not weight-exact.

## Web app files

- `cnn/3d.html` — page, UI panels, sample/Grad-CAM wiring.
- `js/nn/viz.js` — data-driven Three.js renderer (cubes, labels, waveform, edges, Grad-CAM colors).
- `js/nn/infer.js` — TF.js multi-output inference, conv-filter extraction, Grad-CAM.
- `js/three.min.js`, `js/myOrbitControls.js`, `js/stats.min.js`, `js/jquery-*.js`,
  `js/colormaps/myColorMap_dark.js` — vendored libs. TF.js is loaded from a CDN in `cnn/3d.html`.

## Credits

Built on Adam W. Harley, *An Interactive Node-Link Visualization of Convolutional Neural Networks*,
ISVC 2015. Dataset: AudioMNIST (Becker et al.).
