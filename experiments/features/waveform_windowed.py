from audio import wavfile
from musicnet.annotation import dataset_label
from scipy.signal.windows.windows import hann
import numpy as np

def construct_process(data_path: str, label_path: str) -> tuple:

    flen = 1024  # 時間長 64ms  周波数分解能 15.625 Hz
    fshift = 256  # 時間長 16ms

    waveform = wavfile.read(data_path)
    labels = dataset_label.load(label_path)
    waveform = waveform.data

    window = hann(flen)
    nframe = 1 + (len(waveform) - flen) // fshift
    frames = []
    hotvectors = []
    for i in range(nframe):
        fstart = i * fshift
        fend = fstart + flen

        label = labels.frame_mid_pitches(fstart, fend)
        hotvector = dataset_label.list2hotvector(label)
        tup = np.where(hotvector is 1.0)
        if len(tup) > 1:
            continue

        frame = window * waveform[fstart:fend]
        frames.append(frame)
        hotvectors.append(hotvector)

    frames = np.array(frames, dtype=np.float32)
    hotvectors = np.array(hotvectors, dtype=np.float32)
    return frames, hotvectors
