import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as readwav, write as writewav
from json import loads
from os import path
from scipy import signal
import wave
from vosk import Model, KaldiRecognizer, SetLogLevel


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


def plotPeaks(audio, frame_center, frames_start, hnr_frames, peaks_prop, peaks, tt):
    plt.figure(figsize=(15, 5))
    plt.plot(tt, audio, linewidth=1, label="signal")
    plt.xlabel("Time (s)")
    plt.plot(frame_center, hnr_frames, "*", label="HNR")
    ymin, ymax = plt.ylim()
    plt.vlines(tt[frames_start], ymin, ymax, linestyles="dashed")
    # plt.xlim(0,1.5)
    plt.legend()
    plt.show()

    # print(peaks_prop.keys())
    plt.figure()
    plt.plot(frame_center, hnr_frames)

    # mark peaks
    plt.plot(frame_center[peaks], peaks_prop["peak_heights"], "*")
    plt.xlabel("Time(s)")
    plt.ylabel("HNR")
    plt.show()


def preprocess(path_input: str, path_output="audio_preproc", bpfilt=None):
    """Preprocess one audio file

    - mono channel
    - normalize volume
    - bandpass filter

    Parameters
    ----------
    path_input: path to file
    path_output: path to folder for output
    bpfilt: low and high cutoff for filter
    """

    name = path.split(path_input[:-4])[-1]
    print(f"preprocessing {name}")

    if not path_input[-4:] == ".wav":
        raise Exception("not a wav-file")

    Fs, x = readwav(path_input)

    # keep only first channel if stereo
    if len(x.shape) == 2:
        x = x[:, 0]

    if not bpfilt == None:
        # bandpass filter
        fmin, fmax = bpfilt[0], bpfilt[1]
        sos = signal.iirfilter(
            17, #Filter order
            [2 * fmin / Fs, 2 * fmax / Fs],
            rs=60,
            btype="band",
            analog=False,
            ftype="cheby2",
            output="sos",
        )
        x = signal.sosfilt(sos, x)

    # normalize
    x = wavScaler(x)

    if path.exists(path_output):
        writewav(path.join(path_output, name + "_pp.wav"), Fs, x)
    else:
        print("output folder not found")

    return x


def rec_vosk(audio_path: str, model, print_summary=True) -> list[dict]:
    """Recognize speech in a audio file, using a provided vosk-model

    returns: words (list of dicts) contains the word, start, end, conf"""
    wf = wave.open(audio_path, "rb")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    # list of word dictionaries
    results = []

    # recognize speech, vosk
    while True:
        data = wf.readframes(wf.getframerate())
        if len(data) == 0:  # if end
            break
        if rec.AcceptWaveform(data):
            part_result = loads(rec.Result())
            results.append(part_result)

    wf.close()  # close audiofile

    part_result = loads(rec.FinalResult())
    results.append(part_result)

    words = []
    for sentence in results:
        if len(sentence) == 1:
            # sometimes there are bugs in recognition
            # and it returns an empty dictionary
            # {'text': ''}
            continue
        for w in sentence["result"]:
            words.append(w)  # and add it to list

    if print_summary:
        for w in words:
            print_w(w)

    return words


def print_w(w):
    """prints the word, and its information, from a word dictionary"""
    print(
        "{:20} from {:.2f} to {:.2f} sec, confidence: {:.2f}%".format(
            w["word"] + " " + ("-" * (20 - len(w["word"]))),
            w["start"],
            w["end"],
            w["conf"] * 100,
        )
    )
