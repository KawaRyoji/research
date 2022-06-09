from scipy.signal.windows.windows import hann
from scipy.fft import fft
import numpy as np
from audio.wavfile import wavfile
from musicnet.annotation import dataset_label


def construct_process(
    data_path: str, label_path: str, fft_len=2048, flen=1024, fshift=256
) -> tuple:
    waveform = wavfile.read(data_path)
    labels = dataset_label.load(label_path)
    waveform = waveform.data

    # フレームを中点から始めるための処理
    # librosaのデフォルトに合わせて'reflect'にしている
    pad = flen // 2
    waveform = np.pad(waveform, pad, mode="reflect")
    window = hann(flen)

    nframes = 1 + (len(waveform) - flen) // fshift
    frames = []
    hotvectors = []
    for i in range(nframes):
        fstart = i * fshift
        fend = fstart + flen

        frame = waveform[fstart:fend]
        frame = window * frame
        frame = np.abs(fft(frame, n=fft_len))
        frame = frame[: fft_len // 2]

        label = labels.frame_mid_pitches(fstart - pad, fend - pad)
        label = dataset_label.list2hotvector(label)

        frames.append(frame)
        hotvectors.append(label)

    frames = np.array(frames, dtype=np.float32)
    hotvectors = np.array(hotvectors, dtype=np.float32)
    return frames, hotvectors
