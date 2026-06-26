"""Bridge the trained Keras model to the web app.

Produces, under js/nn/:
  - tfjs_model/        TensorFlow.js model (model.json + weight shards)
  - layout.json        per-layer spec + 3D node positions for the visualization
  - samples.json       a handful of preprocessed clips (waveform + WAV audio)

Runs against the trained model if present; otherwise builds an untrained model so
the web demo is renderable before training (predictions will be meaningless).
Likewise, samples come from the real dataset if available, else are synthesized.
"""
import base64
import io
import json
import os
import struct
import subprocess
import sys

import numpy as np

import config

# Visualization layer order: each maps to a named Keras layer output.
# type drives geometry/edges in the renderer.
VIZ_LAYERS = [
    {"name": "waveform", "label": "Input waveform", "type": "input"},
    {"name": "conv1", "label": "Convolution 1", "type": "conv"},
    {"name": "pool1", "label": "Pooling 1", "type": "pool"},
    {"name": "conv2", "label": "Convolution 2", "type": "conv"},
    {"name": "pool2", "label": "Pooling 2", "type": "pool"},
    {"name": "conv3", "label": "Convolution 3", "type": "conv"},
    {"name": "pool3", "label": "Pooling 3", "type": "pool"},
    {"name": "conv4", "label": "Convolution 4", "type": "conv"},
    {"name": "pool4", "label": "Pooling 4", "type": "pool"},
    {"name": "dense1", "label": "Fully-connected 1", "type": "dense"},
    {"name": "output", "label": "Output", "type": "dense"},
]


def get_model():
    from tensorflow import keras
    if os.path.exists(config.MODEL_PATH):
        print("Loading trained model %s" % config.MODEL_PATH)
        return keras.models.load_model(config.MODEL_PATH, compile=False), True
    print("No trained model found — building an untrained model for the demo.")
    import train
    return train.build_model(), False


def display_len(actual_len, layer_type):
    cap = config.INPUT_DISPLAY_LEN if layer_type == "input" else config.DISPLAY_LEN_CAP
    return min(actual_len, cap)


def build_layout(model):
    """Return (layers_spec, nodes) using each layer's output shape."""
    layers_spec = []
    nodes = []
    name_to_layer = {l.name: l for l in model.layers}

    for layer_num, viz in enumerate(VIZ_LAYERS):
        layer = name_to_layer[viz["name"]]
        shape = layer.output.shape  # (None, length, channels) or (None, units)
        if len(shape) == 3:
            length, channels = int(shape[1]), int(shape[2])
        else:
            length, channels = int(shape[1]), 1  # dense -> line of units

        dlen = display_len(length, viz["type"])
        kernel = int(getattr(layer, "kernel_size", [0])[0]) if hasattr(layer, "kernel_size") else 0
        strides = int(getattr(layer, "strides", [1])[0]) if hasattr(layer, "strides") else 1
        pool = int(getattr(layer, "pool_size", [1])[0]) if hasattr(layer, "pool_size") else 0

        layers_spec.append({
            "name": viz["name"], "label": viz["label"], "type": viz["type"],
            "length": length, "channels": channels, "displayLen": dlen,
            "kernel": kernel, "strides": strides, "pool": pool,
        })

        y = layer_num * config.LAYER_GAP
        for t in range(dlen):                 # time-major, then channel
            x = (t - (dlen - 1) / 2.0) * config.X_SPACING
            for c in range(channels):
                z = (c - (channels - 1) / 2.0) * config.Z_SPACING
                nodes.append({"x": round(x, 2), "y": round(y, 2),
                              "z": round(z, 2), "layerNum": layer_num})

    return layers_spec, nodes


def wav_base64(waveform, sr=config.SAMPLE_RATE):
    """Encode a float32 [-1,1] waveform as a base64 16-bit PCM WAV (data URI body)."""
    pcm = np.clip(waveform, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2").tobytes()
    n = len(pcm)
    header = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    header += b"data" + struct.pack("<I", n)
    return base64.b64encode(header + pcm).decode("ascii")


def gather_samples():
    if os.path.exists(config.DATA_NPZ):
        d = np.load(config.DATA_NPZ)
        X, y = d["X_test"], d["y_test"]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(config.N_WEB_SAMPLES, len(X)), replace=False)
        return [(X[i, :, 0].astype(np.float32), int(y[i])) for i in idx]

    print("No dataset — synthesizing placeholder waveforms so the demo renders.")
    samples = []
    t = np.linspace(0, 1, config.INPUT_LEN, endpoint=False)
    for digit in range(config.N_WEB_SAMPLES):
        f = 120 + 60 * (digit % config.N_CLASSES)
        w = 0.6 * np.sin(2 * np.pi * f * t) * np.exp(-3 * t)
        w += 0.05 * np.random.default_rng(digit).standard_normal(config.INPUT_LEN)
        samples.append((w.astype(np.float32), digit % config.N_CLASSES))
    return samples


def write_samples(samples):
    out = []
    for wave, label in samples:
        out.append({
            "label": label,
            "waveform": [round(float(v), 5) for v in wave.tolist()],
            "wav": wav_base64(wave),
        })
    with open(config.SAMPLES_JSON, "w") as f:
        json.dump({"sampleRate": config.SAMPLE_RATE, "inputLen": config.INPUT_LEN,
                   "samples": out}, f)
    print("Wrote %s (%d samples)" % (config.SAMPLES_JSON, len(out)))


def convert_tfjs(model):
    os.makedirs(config.TFJS_MODEL_DIR, exist_ok=True)
    try:
        import tensorflowjs  # noqa: F401
        cmd = [
            sys.executable, "-m", "tensorflowjs.converters.converter",
            "--input_format=keras", config.MODEL_PATH, config.TFJS_MODEL_DIR,
        ]
        print("Converting to TF.js via tensorflowjs: %s" % " ".join(cmd))
        subprocess.check_call(cmd)
    except Exception as e:
        print("tensorflowjs unavailable/failed (%s) — using built-in exporter." % e)
        import keras_to_tfjs
        keras_to_tfjs.export(model, config.TFJS_MODEL_DIR)
        print("Wrote TF.js model to %s" % config.TFJS_MODEL_DIR)


def main():
    os.makedirs(config.WEB_NN_DIR, exist_ok=True)
    model, trained = get_model()

    layers_spec, nodes = build_layout(model)
    with open(config.LAYOUT_JSON, "w") as f:
        json.dump({"layers": layers_spec, "nodes": nodes,
                   "inputLen": config.INPUT_LEN, "sampleRate": config.SAMPLE_RATE}, f)
    print("Wrote %s (%d nodes, %d layers)" % (config.LAYOUT_JSON, len(nodes), len(layers_spec)))

    write_samples(gather_samples())

    if trained:
        convert_tfjs(model)
    else:
        # Save the untrained model so the converter has something to read.
        os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
        model.save(config.MODEL_PATH)
        convert_tfjs(model)
        print("NOTE: exported an UNTRAINED model — run model/train.py for real predictions.")


if __name__ == "__main__":
    main()
