"""Train a 1D CNN on the preprocessed AudioMNIST raw-waveform dataset.

The architecture is intentionally small (few channels, aggressive striding/pooling)
so the resulting per-layer activation maps stay light enough to render as 3D nodes.
Layers are named so export_web.py can map activations to visualization layers.
"""
import os
import sys

import numpy as np

import config


def build_model():
    from tensorflow import keras
    from tensorflow.keras import layers

    # Uniform 7-tap kernels throughout. The first conv keeps stride 4 to
    # downsample the raw waveform; depth + max-pooling grow the effective
    # receptive field. Channel counts are kept small so the per-layer
    # activations stay renderable as 3D nodes.
    inp = keras.Input(shape=(config.INPUT_LEN, 1), name="waveform")
    x = layers.Conv1D(8, 7, strides=4, padding="same", activation="relu", name="conv1")(inp)
    x = layers.MaxPooling1D(4, name="pool1")(x)
    x = layers.Conv1D(16, 7, padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling1D(4, name="pool2")(x)
    x = layers.Conv1D(32, 7, padding="same", activation="relu", name="conv3")(x)
    x = layers.MaxPooling1D(4, name="pool3")(x)
    x = layers.Conv1D(32, 7, padding="same", activation="relu", name="conv4")(x)
    x = layers.MaxPooling1D(4, name="pool4")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    out = layers.Dense(config.N_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inp, out, name="audiomnist_1dcnn")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    if not os.path.exists(config.DATA_NPZ):
        sys.exit("Dataset not found at %s — run model/preprocess.py first." % config.DATA_NPZ)

    d = np.load(config.DATA_NPZ)
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]
    print("train=%s  test=%s" % (X_train.shape, X_test.shape))

    model = build_model()
    model.summary()

    epochs = int(os.environ.get("EPOCHS", "25"))
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs, batch_size=64,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy: %.4f" % acc)

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    model.save(config.MODEL_PATH)
    print("Saved %s" % config.MODEL_PATH)


if __name__ == "__main__":
    main()
