"""Preprocess AudioMNIST wavs into a fixed-length 8kHz raw-waveform dataset.

Loads every {digit}_{speaker}_{rep}.wav, resamples 48k -> 8k, pads/trims to
INPUT_LEN samples, peak-normalizes, and saves a stratified train/test split.
"""
import glob
import os
import sys

import numpy as np

import config


def load_wav(path):
    import librosa
    y, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    # pad / trim to fixed length
    if len(y) < config.INPUT_LEN:
        y = np.pad(y, (0, config.INPUT_LEN - len(y)))
    else:
        y = y[: config.INPUT_LEN]
    # peak normalize
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def main():
    if not os.path.isdir(config.DATA_WAV_DIR):
        sys.exit(
            "AudioMNIST data not found at %s\n"
            "Run model/download_data.sh first." % config.DATA_WAV_DIR
        )

    wavs = sorted(glob.glob(os.path.join(config.DATA_WAV_DIR, "**", "*.wav"), recursive=True))
    if not wavs:
        sys.exit("No .wav files found under %s — run model/download_data.sh." % config.DATA_WAV_DIR)

    print("Found %d wavs. Preprocessing..." % len(wavs))
    X, y, speakers = [], [], []
    for i, path in enumerate(wavs):
        name = os.path.splitext(os.path.basename(path))[0]  # e.g. "0_01_0"
        parts = name.split("_")
        digit = int(parts[0])
        speaker = parts[1] if len(parts) > 1 else "0"
        X.append(load_wav(path))
        y.append(digit)
        speakers.append(speaker)
        if (i + 1) % 2000 == 0:
            print("  %d/%d" % (i + 1, len(wavs)))

    X = np.stack(X)[..., np.newaxis]  # (N, INPUT_LEN, 1)
    y = np.array(y, dtype=np.int64)
    speakers = np.array(speakers)

    # Speaker-disjoint split: hold out ~15% of speakers for test.
    uniq = np.unique(speakers)
    rng = np.random.default_rng(42)
    rng.shuffle(uniq)
    n_test = max(1, int(round(0.15 * len(uniq))))
    test_speakers = set(uniq[:n_test])
    test_mask = np.array([s in test_speakers for s in speakers])

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    np.savez_compressed(
        config.DATA_NPZ,
        X_train=X[~test_mask], y_train=y[~test_mask],
        X_test=X[test_mask], y_test=y[test_mask],
    )
    print("Saved %s  (train=%d, test=%d)" % (config.DATA_NPZ, (~test_mask).sum(), test_mask.sum()))


if __name__ == "__main__":
    main()
