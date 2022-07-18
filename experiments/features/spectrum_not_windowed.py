from typing import Tuple
import numpy as np
from scipy.fft import fft

from audio.wavfile import wavfile
from musicnet.annotation import dataset_label


def construct_process(
    data_path: str, label_path: str, fft_len=2048, flen=1024, fshift=256
) -> Tuple[np.ndarray, np.ndarray]:
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

        frame = waveform[fstart:fend]
        frame = np.abs(fft(frame, n=fft_len))
        frame = frame[: fft_len // 2]

        frames.append(frame)
        hotvectors.append(hotvector)

    frames = np.array(frames, dtype=np.float32)
    hotvectors = np.array(hotvectors, dtype=np.float32)
    return frames, hotvectors
