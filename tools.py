import numpy as np


def split_frames(x, fl, Fs, overlap=0):
    """Splits the signal ``x`` into frames of length ``fl``."""

    # frame start indices, discard short last frame
    f_start = np.arange(0, len(x), fl - overlap)[:-1]
    # list of frames
    frames = [x[s : s + fl] for s in f_start]

    print(f"frame length    : {fl} samples")
    print(f"frame length    : {np.round(fl/Fs,3)} seconds")
    print(f"number of frames: {len(f_start)}")

    return frames, f_start

def wavScaler(x):
    """Scales a signal to wavfile integer range"""
    return np.int16(x / np.max(np.abs(x)) * np.iinfo(np.int16).max)

def stitch_frames(frames, fade_pow, padding="?"):

    frames = [wavScaler(f) for f in frames]
    return np.concatenate(frames)


def fade_sound(x, pow=0.2):
    """Fades start and end"""
    w = np.hamming(len(x)) ** pow
    return x * w