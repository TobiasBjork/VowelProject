import wave
from json import loads
from os import path

import librosa as lib
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, ndimage, signal
from scipy.io.wavfile import read as readwav
from scipy.io.wavfile import write as writewav
from Signal_Analysis.features.signal import get_F_0, get_HNR
from vosk import KaldiRecognizer, Model, SetLogLevel
from sklearn import preprocessing, ensemble

from folderFunctions import *

# constants
VOWELS_SV = ("e", "y", "u", "i", "o", "å", "a", "ö", "ä")


def vol(x):
    return np.mean(np.abs(x))


def vol_db(x, ref=1):
    return 10 * np.log10(np.mean(x**2) / ref)


def envelope(s, dmax=1, smoothing=10):
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    # global max of dmax-chunks of locals max
    lmax = lmax[
        [i + np.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]
    ]
    tt = np.arange(len(s))
    interp = interpolate.CubicSpline(tt[lmax], s[lmax])
    env = interp(tt)
    env = ndimage.gaussian_filter1d(env, smoothing)

    return env


def split_frames(x, fl, Fs, overlap=0, vol_thr=0, print_info=False):
    """Splits the signal ``x`` into frames of length ``fl``.
    ## Returns
    frames: list[ndarray]
    f_start: ndarray"""

    # frame start indices
    f_start = np.arange(0, len(x), fl - overlap)
    # list of frames
    frames = [x[s : s + fl] for s in f_start]
    frames = [f * (vol(f) > vol_thr) for f in frames]

    if print_info:
        print(f"frame length    : {fl} samples")
        print(f"frame length    : {np.round(fl/Fs,3)} seconds")
        print(f"between frames  : {np.round((fl-overlap)/Fs,3)} seconds")

        print(f"number of frames: {len(f_start)}")

    return frames, f_start


def wavScaler(x):
    """Scales a signal to wavfile integer range"""
    if np.max(np.abs(x)) == 0:
        return np.int16(x)
    return np.int16(x / np.max(np.abs(x)) * np.iinfo(np.int16).max)


def stitch_frames(frames, fade_pow=0.0, padding=0):
    """concatenate frames together, with optional fading and padding (samples silence) between frames.
    Also scales to wav-integer"""

    if fade_pow > 0:
        frames = [wavScaler(fade_sound(f, fade_pow)) for f in frames]
    else:
        frames = [wavScaler(f) for f in frames]

    # add padding zeros after each frame
    frames = [np.concatenate((f, np.zeros(int(padding)))) for f in frames]

    return np.concatenate(frames).astype(np.int16)


def fade_sound(x, pow=0.0):
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
        if x is True:
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
    plt.figure(figsize=(15, 10))
    plt.subplot(211)
    plt.plot(tt, audio, linewidth=1, label="signal")
    plt.xlabel("Time (s)")
    plt.plot(frame_center, hnr_frames, "*", label="HNR")
    ymin, ymax = plt.ylim()
    plt.vlines(tt[frames_start], ymin, ymax, linestyles="dashed")
    # plt.xlim(0,1.5)
    plt.legend()

    # print(peaks_prop.keys())
    plt.subplot(212)
    plt.plot(frame_center, hnr_frames)

    # mark peaks
    plt.plot(frame_center[peaks], peaks_prop["peak_heights"], "*")
    plt.xlabel("Time(s)")
    plt.ylabel("HNR")
    plt.tight_layout()
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
    bpfilt: low and high cutoff for filter. Default none
    """

    name = path.split(path_input[:-4])[-1]
    print(f"preprocessing {name}")

    if not path_input[-4:] == ".wav":
        raise Exception("not a wav-file")

    Fs, x = readwav(path_input)

    # keep only first channel if stereo
    if len(x.shape) == 2:
        x = x[:, 0]

    if bpfilt is not None:
        # bandpass filter
        fmin, fmax = bpfilt[0], bpfilt[1]
        sos = signal.iirfilter(
            17,  # Filter order
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
        for i, w in enumerate(words):
            print_w(i, w)

    return words


def print_w(i, w):
    """prints the word, and its information, from a word dictionary"""
    print(
        i,
        "{:20} from {:.4f} to {:.4f} sec, confidence: {:.2f}%".format(
            w["word"] + " " + ("-" * (20 - len(w["word"]))),
            w["start"],
            w["end"],
            w["conf"] * 100,
        ),
    )


def checkVowels(word, vowels):
    """Returns list of found vowels in a word"""
    foundVowels = [letter for letter in word if letter in vowels]
    return foundVowels


def segment_by_words(list_of_words, audio, Fs, vowel_set, min_conf=1, signal_pad=0):
    """splits an audio into segments, by vosk words.
    ## Parameters:

    zero_padd: boolean
        Add one zero on each side of segment

    signal_pad: int
        add extra seconds from signal on each side of segment

    ## Returns
    segments: ndarray
    vowels_per_segment: list
    s_start: list"""
    if not type(list_of_words[0] == dict):
        raise Exception("expects a dict for each word")
    segments = []
    vowels_per_segment = []
    s_start = []
    for word in list_of_words:
        vowels_per_segment.append(checkVowels(word["word"].lower(), vowel_set))
        start = round(
            max(word["start"] * Fs - signal_pad * Fs, 0)
        )  # start of word (sample)
        end = round(min(word["end"] * Fs + signal_pad * Fs, len(audio)))  # end of word.
        a = audio[start:end]

        segments.append(a)
        s_start.append(start)

        # adding word to the list
    return segments, vowels_per_segment, s_start


def checkIfWhite(signal, wNoiseRatio=0.8):
    """Checks if input signal noisy
    ## returns
    white: bool"""
    sum0, sum1 = 0, 0
    for i in range(1, len(signal)):
        sum0 += abs(signal[i])
        sum1 += abs(signal[i] - signal[i - 1])
    return sum1 / sum0 > wNoiseRatio


def HNR_peaks(audio, Fs, n_peaks=-1, plotit=False):
    tt = np.linspace(0, len(audio) / Fs, len(audio))
    fl = int(0.08 * Fs)
    frames, frames_start = split_frames(audio, fl, Fs, overlap=int(6 * fl / 8))
    tt_frames_center = tt[frames_start] + int(fl / 2) / Fs
    hnr_frames = []
    for f in frames:
        hnr_frames.append(get_HNR(f, Fs))
    min_h = max(hnr_frames) / 10  # osäker grej
    peaks, peaks_prop = signal.find_peaks(hnr_frames, height=min_h)
    order = np.argsort(-peaks_prop["peak_heights"])
    peaks = [peaks[i] for i in order][:n_peaks]

    # peaks = peaks[peaks > np.sorted(peaks, reverse=True)[n_peaks]]

    for k in peaks_prop.keys():
        peaks_prop[k] = [peaks_prop[k][i] for i in order][:n_peaks]

    # width of peak at half max
    width = signal.peak_widths(hnr_frames, peaks, rel_height=0.5)[0]
    # width = 1 * np.ones(len(peaks))
    peak_sounds = []
    for i in range(len(width)):
        print(width[i])
        start = int((Fs * tt_frames_center[peaks[i]] - width[i] / 2 * fl))
        end = int((Fs * tt_frames_center[peaks[i]] + width[i] / 2 * fl))

        peak_sounds.append(audio[max(0, start) : min(end, len(audio))])

    if plotit:
        plt.figure(figsize=(15, 5))
        plt.plot(tt, audio)
        plt.vlines(tt[start], *plt.ylim(), colors="r")
        plt.vlines(tt[end], *plt.ylim(), colors="r")
        plotPeaks(
            audio, tt_frames_center, frames_start, hnr_frames, peaks_prop, peaks, tt
        )

    return frames, peaks_prop, peaks, peak_sounds


def HNR_short(frames, Fs, n_peaks=-1, min_dist=True):
    """get frame index for peaks and hnr per frame"""
    hnr_frames = np.array([get_HNR(f, Fs) for f in frames])

    if min_dist:
        min_distance = max(len(frames) / (n_peaks + 2), 1)  # frames
    else:
        min_distance = 1
    peaks, peaks_prop = signal.find_peaks(
        hnr_frames,
        height=0.1 * max(hnr_frames),
        distance=min_distance,
    )
    order = (-peaks_prop["peak_heights"]).argsort()  # order
    peaks = peaks[order[:n_peaks]]  # sort and truncate
    peaks = np.sort(peaks)  # restore order
    return peaks, hnr_frames


def extractVowels(segments, vowels_segments, Fs, language, id):
    for i in range(len(segments)):
        frames, peaks_prop, peaks, peaks_sounds = HNR_peaks(segments[i], Fs)
        if len(peaks) == len(vowels_segments[i]):
            for j in range(len(peaks)):
                updateFolder(language, peaks_sounds[j], vowels_segments[i][j], id, Fs)


def get_mfcc(x, Fs, n=50, normalize=True):
    """Compute n first MFCCoefficients,
    for a list of segments it returns coefficient for every (normalized) segment"""
    if isinstance(x, list):
        return [
            np.mean(
                lib.feature.mfcc(y=normalize_std(xi), sr=Fs, n_mfcc=n, n_fft=512).T,
                axis=0,
            )
            for xi in x
        ]
    else:
        if normalize:
            return np.mean(
                lib.feature.mfcc(y=normalize_std(x), sr=Fs, n_mfcc=n, n_fft=512).T,
                axis=0,
            )
        else:
            return np.mean(lib.feature.mfcc(y=x, sr=Fs, n_mfcc=n, n_fft=512).T, axis=0)


def normalize_std(x):
    """Normalize a signal, returns zero for zero signal"""
    if np.ptp(x) == 0:
        return x
    else:
        return x / np.std(x)


def julgran(words, audio, Fs, fl, add_context=False, long_frame=False):
    grouped_frames = {v: {} for v in VOWELS_SV}

    # initialize lists
    for v in grouped_frames.keys():
        grouped_frames[v]["frame"] = []
        grouped_frames[v]["start"] = []
        grouped_frames[v]["stop"] = []

    segments, vowels_per_segment, s_starts = segment_by_words(
        words, audio, Fs, VOWELS_SV, signal_pad=0
    )
    for w, segment, vowels, start_segment in zip(
        words, segments, vowels_per_segment, s_starts
    ):
        if w["conf"] >= 1:
            # zero padding
            if long_frame:
                segment = np.concatenate((np.zeros(3 * fl), segment, np.zeros(3 * fl)))

                frames, f_start = split_frames(
                    segment, 3 * fl, Fs, vol_thr=0, overlap=int(2 * fl)
                )
                peak_frames, hnr_frames = HNR_short(
                    frames, Fs, len(vowels), min_dist=False
                )
            else:
                segment = np.concatenate((np.zeros(fl), segment, np.zeros(fl)))

                frames, f_start = split_frames(
                    segment, fl, Fs, vol_thr=0, overlap=int(0)
                )
                peak_frames, hnr_frames = HNR_short(frames, Fs, len(vowels))

            # Check all vowels in word before keeping frames
            keep_word = False
            if len(peak_frames) == len(vowels):
                keep_word = True
                for i, v in enumerate(vowels):
                    frame = frames[peak_frames[i]]
                    noise_check = not checkIfWhite(frame, wNoiseRatio=0.5)
                    vol_check = vol_db(frame) > 45
                    zero_check = (
                        np.sum(abs(frame) < 0.1 * max(frame)) / len(frame) < 0.5
                    )
                    if not (noise_check and vol_check and zero_check):
                        keep_word = False

            if not keep_word:
                print("trash", w["word"])
            if keep_word:
                print("keep", w["word"])

                for i, v in enumerate(vowels):
                    if add_context:
                        grouped_frames[v]["frame"].append(
                            stitch_frames(
                                frames[
                                    max(peak_frames[i] - 2, 0) : min(
                                        peak_frames[i] + 2, len(frames)
                                    )
                                ]
                            )
                        )
                    else:
                        if long_frame:
                            f_long = frames[peak_frames[i]]

                            grouped_frames[v]["frame"].append(
                                f_long[
                                    int(len(f_long) / 2 - fl / 2) : int(
                                        len(f_long) / 2 + fl / 2
                                    )
                                ]
                            )
                            start_vowel = (
                                start_segment
                                + f_start[peak_frames[i]]
                                - 3 * fl
                                + len(f_long) / 2
                                - fl / 2
                            )

                        else:
                            grouped_frames[v]["frame"].append(frames[peak_frames[i]])
                            # start and stop (of real frame) (-fl compensates zeropadding)
                            start_vowel = start_segment + f_start[peak_frames[i]] - fl

                        grouped_frames[v]["start"].append(start_vowel)
                        grouped_frames[v]["stop"].append(start_vowel + fl)

    return grouped_frames


def outlier_filter(grouped_frames, Fs):
    grouped_features = {v: [] for v in VOWELS_SV}
    frames_inlier = {v: [] for v in VOWELS_SV}

    for v in VOWELS_SV:
        for f in grouped_frames[v]["frame"]:
            mfcc = get_mfcc(f, Fs, n=5, normalize=True)
            feat_vec = np.concatenate((mfcc, f))
            grouped_features[v].append(feat_vec)

        X = np.array(grouped_features[v])

        sc_X = preprocessing.StandardScaler()
        X = sc_X.fit_transform(X)
        clf = ensemble.IsolationForest(random_state=0)
        clf.fit(X)

        inliers = clf.predict(X) > 0
        frames_inlier[v] = {
            k: [grouped_frames[v][k][i] for i in range(len(inliers)) if inliers[i]]
            for k in grouped_frames[v].keys()
        }
        print(f"inliers ({v}): {np.around(100*sum(inliers)/len(X))} %")

    return frames_inlier


def get_start_stop_seconds(grouped_frames, fl, Fs):
    """Not a good method"""
    starts_all = []
    for v in grouped_frames.keys():
        starts_all.extend(grouped_frames[v]["start"])
    starts_all = np.array(starts_all)

    print("total found vowels:", len(starts_all))
    print("unique start points:", len(np.unique(starts_all)))
    starts_all = np.sort(starts_all)
    stops_all = starts_all + fl
    starts_all_seconds = starts_all / Fs
    stops_all_seconds = stops_all / Fs
    return starts_all_seconds, stops_all_seconds

def score_vs_labels(starts, stops, labels_df):
    """computes precision and recall, by comparing starts and stops with labels tmin, tmax
    Does not consider which vowel it actually is"""
    included = 0
    for start, stop in zip(starts, stops):
        included += labels_df.apply(
            lambda x: (start > x.tmin) and (stop < x.tmax), axis=1
        ).sum()

    print(included)

    prec = included / len(starts)
    reca = included / len(labels_df)

    print("Assuming Praat perfect and all vowels correctly classified:")
    print("precision:", prec)
    print("recall:", included / len(labels_df))
