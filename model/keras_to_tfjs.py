"""Minimal, dependency-free Keras -> TensorFlow.js LayersModel exporter.

Used as a fallback when the `tensorflowjs` package is unavailable. Writes the
standard layers-model format (model.json + a single weight shard) that
tf.loadLayersModel() consumes: model topology from model.to_json() and weights
concatenated as little-endian float32 in model.weights order.
"""
import json
import os

import numpy as np


def export(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    topo = json.loads(model.to_json())  # {class_name, config, keras_version, backend}
    model_topology = {
        "keras_version": topo.get("keras_version", "2.13.1"),
        "backend": topo.get("backend", "tensorflow"),
        "model_config": {"class_name": topo["class_name"], "config": topo["config"]},
    }

    weight_entries = []
    buffers = []
    for w in model.weights:
        name = w.name
        if name.endswith(":0"):
            name = name[:-2]
        arr = np.asarray(w.numpy(), dtype="<f4")
        buffers.append(arr.tobytes())
        weight_entries.append({"name": name, "shape": list(arr.shape), "dtype": "float32"})

    shard = "group1-shard1of1.bin"
    with open(os.path.join(out_dir, shard), "wb") as f:
        f.write(b"".join(buffers))

    model_json = {
        "format": "layers-model",
        "generatedBy": "keras " + model_topology["keras_version"],
        "convertedBy": "keras_to_tfjs.py",
        "modelTopology": model_topology,
        "weightsManifest": [{"paths": [shard], "weights": weight_entries}],
    }
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(model_json, f)
    return out_dir
