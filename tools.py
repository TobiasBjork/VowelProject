import numpy as np


def split_frames(x, fl, Fs, overlap=0):
    """Splits the signal ``x`` into frames of length ``fl``."""

    f_start = np.arange(0, len(x), fl - overlap)[:-1]
    frames = np.stack([x[s : s + fl] for s in f_start])
    index = np.arange(frames.size).reshape(frames.shape)

    print(f"frame length    : {fl} samples")
    print(f"frame length    : {np.round(fl/Fs,3)} seconds")
    print(f"number of frames: {len(f_start)}")

    return frames, index

def wavScaler(x):
    """Scales a signal to wavfile integer range"""
    return np.int16(x / np.max(np.abs(x)) * np.iinfo(np.int16).max)

def stitch_frames(frames, fade, padding = "?"):
    """TODO """