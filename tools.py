import numpy as np


def split_frames(x, fl, Fs, overlap=0):
    """Splits the signal ``x`` into frames of length ``fl``."""

    # frame start indices, discard short last frame
    f_start = np.arange(0, len(x), fl - overlap)[:-1]
    # list of frames
    frames = [x[s : s + fl] for s in f_start]

    print(f"frame length    : {fl} samples")
    print(f"frame length    : {np.round(fl/Fs,3)} seconds")
    print(f"between frames  : {np.round((fl-overlap)/Fs,3)} seconds")

    print(f"number of frames: {len(f_start)}")

    return frames, f_start


def wavScaler(x):
    """Scales a signal to wavfile integer range"""
    return np.int16(x / np.max(np.abs(x)) * np.iinfo(np.int16).max)


def stitch_frames(frames, fade_pow=0, padding=0):
    """concatenate frames together, with optional fading and padding (silence) between frames.
    Also scales to wav-integer"""

    if fade_pow > 0:
        frames = [wavScaler(fade_sound(f, fade_pow)) for f in frames]
    else:
        frames = [wavScaler(f) for f in frames]

    # add padding zeros after each frame
    frames = [np.concatenate((f, np.zeros(int(padding)))) for f in frames]

    return np.concatenate(frames).astype(np.int16)


def fade_sound(x, pow=0):
    """Fades start and end of signal"""
    w = np.hamming(len(x)) ** pow
    return x * w


def movmean_peak(sequence, lag=5, thr=1, peak_infl=0.1, duration=1):
    """finds peaks that differ from moving mean

    Parameters
    ----------
    a : array_like
        signal to find peaks in

    lag:
        length ov moving mean/std

    thr:
        number of standard deviations for peak

    peak_infl: between 0 and 1
        How strongly peaks influence computed mean/std

    duration:
        minimum length (in samples) of a peak

    Returns
    ----------
    peak_idx:
        Indices of peaks
    """
    is_peak = [
        False,
    ] * lag
    processed_seq = list(sequence)[:lag].copy()

    for i in range(lag, len(sequence)):
        y = sequence[i]
        avg = np.mean(processed_seq[i - lag : i])
        std = np.std(processed_seq[i - lag : i])

        if y - avg > std * thr:
            is_peak.append(True)
            # calculate next step from peak or last value?
            processed_seq.append(peak_infl * y + (1 - peak_infl) * processed_seq[i - 1])
        else:
            is_peak.append(False)
            processed_seq.append(y)

    # length of current peak
    in_len = 0
    for i, x in enumerate(is_peak):
        if x == True:
            in_len += 1
        else:
            if in_len < duration:
                # remove previous peak if too short
                is_peak[i - in_len : i] = [
                    False,
                ] * in_len
            in_len = 0

    return is_peak

def binary_start_stop(sequence):
    """
    Return start (inclusive) and endpoints (exclusive) for nonzero subsequences
    """

    a = sequence.copy()
    starts = []
    stops = []
    if sequence[0] > 0:
        starts.append(0)

    for i in range(1, len(a) - 1):
        if a[i] > 0 and a[i - 1] == 0:
            starts.append(i)
        if a[i] == 0 and a[i - 1] > 0:
            stops.append(i)

    return starts, stops
