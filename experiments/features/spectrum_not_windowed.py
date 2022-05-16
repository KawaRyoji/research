import numpy as np
from scipy.fft import fft

from audio.wavfile import wavfile
from musicnet.annotation import dataset_label

def construct_process(data_path: str, label_path: str) -> tuple:

    fft_len = 2048  # 分解能 8Hz
    flen = 1024  # 時間長 64ms  周波数分解能 15.625 Hz
    fshift = 256  # 時間長 16ms
    waveform = wavfile.read(data_path)
    labels = dataset_label.load(label_path)
    waveform = waveform.data

    pad = flen // 2
    waveform = np.pad(waveform, pad, mode="reflect")
    nframe = 1 + (len(waveform) - flen) // fshift
    frames = []
    hotvectors = []
    for i in range(nframe):
        fstart = i * fshift
        fend = fstart + flen

        label = labels.frame_mid_pitches(fstart - pad, fend - pad)
        hotvector = dataset_label.list2hotvector(label)
        tup = np.where(hotvector is 1.0)
        if len(tup) > 1:
            continue

        frame = waveform[fstart:fend]
        frame = np.abs(fft(frame, n=fft_len))
        frame = frame[: fft_len // 2]

        frames.append(frame)
        hotvectors.append(hotvector)

    frames = np.array(frames, dtype=np.float32)
    hotvectors = np.array(hotvectors, dtype=np.float32)
    return frames, hotvectors