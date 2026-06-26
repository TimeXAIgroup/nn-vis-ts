"""Shared configuration for the AudioMNIST 1D-CNN pipeline."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Data locations
DATA_WAV_DIR = os.path.join(HERE, "data", "AudioMNIST", "data")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
DATA_NPZ = os.path.join(ARTIFACTS_DIR, "data.npz")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.keras")

# Web export targets (the app lives at the repo root)
WEB_NN_DIR = os.path.join(REPO_ROOT, "js", "nn")
TFJS_MODEL_DIR = os.path.join(WEB_NN_DIR, "tfjs_model")
LAYOUT_JSON = os.path.join(WEB_NN_DIR, "layout.json")
SAMPLES_JSON = os.path.join(WEB_NN_DIR, "samples.json")

# Audio preprocessing
SAMPLE_RATE = 8000        # downsample target (AudioMNIST native is 48kHz)
INPUT_LEN = 8000          # fixed window = 1 second @ 8kHz
N_CLASSES = 10

# How many sample clips to bundle into the web app
N_WEB_SAMPLES = 12

# Visualization display caps. Early 1D layers are too long to draw one cube per
# activation, so each layer's time axis is evenly subsampled to at most this many
# positions for rendering. infer.js subsamples the real activation tensor to match.
DISPLAY_LEN_CAP = 48      # max time positions shown for conv/pool layers
INPUT_DISPLAY_LEN = 200   # time positions shown for the raw waveform input layer

# 3D layout spacing
LAYER_GAP = 95.0          # vertical gap between layers (y)
X_SPACING = 9.0           # spacing along the time axis
Z_SPACING = 16.0          # spacing along the channel axis
