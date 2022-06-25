import librosa
import numpy as np

from audio.wavfile import wavfile
from musicnet.annotation import dataset_label
from scipy.signal.windows.windows import hann


def construct_process(
    data_path: str, label_path: str, fft_len=2048, flen=1024, fshift=256, time_len=16
):
    waveform = wavfile.read(data_path)
    labels = dataset_label.load(label_path)
    waveform = waveform.data

    pad = flen // 2
    waveform = np.pad(waveform, pad, mode="reflect")
    window = hann(flen)

    nframe = 1 + (len(waveform) - flen) // fshift
    nspec = nframe // time_len
    
    spectrograms = []
    hotvectors = []
    
    
    librosa.stft()
