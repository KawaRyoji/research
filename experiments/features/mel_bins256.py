import numpy as np
from scipy.signal.windows.windows import hann
from scipy.fft import fft
from audio.wavfile import wavfile
from musicnet.annotation import dataset_label
from librosa.filters import mel
from util.calc import square

def construct_process(data_path: str, label_path: str) -> tuple:
    sampling_freq = 16000
    fft_len = 2048  # 分解能 8Hz
    flen = 1024  # 時間長 64ms
    fshift = 256  # 時間長 16ms
    waveform = wavfile.read(data_path)
    labels = dataset_label.load(label_path)
    waveform = waveform.data

    # フレームを中点から始めるための処理
    # librosaのデフォルトに合わせて'reflect'にしている
    waveform = np.pad(waveform, flen // 2, mode="reflect")
    window = hann(flen)

    nframes = 1 + (len(waveform) - flen) // fshift
    frames = []
    hotvectors = []
    for i in range(nframes):
        fstart = i * fshift
        fend = fstart + flen

        frame = waveform[fstart:fend]
        frame = window * frame
        frame = square(fft(frame, n=fft_len))
        frame = frame[: fft_len // 2]

        label = labels.frame_mid_pitches(fstart, fend)
        label = dataset_label.list2hotvector(label)

        frames.append(frame)
        hotvectors.append(label)

    frames = np.array(frames, dtype=np.float32)
    hotvectors = np.array(hotvectors, dtype=np.float32)
    
    # メル変換
    filter = mel(sampling_freq, fft_len, n_mels=256)
    filter = filter[:, : fft_len // 2]
    frames = np.dot(filter, frames.T).T
    
    return frames, hotvectors
