from audio.wavfile import wavfile
from musicnet.annotation import dataset_label
import numpy as np

def construct_process(data_path: str, label_path: str) -> tuple:
    
    flen = 1024
    fshift = 256

    waveform = wavfile.read(data_path)
    labels = dataset_label.load(label_path)
    waveform = waveform.data

    nframe = 1 + (len(waveform) - flen) // fshift
    frames = []
    hotvectors = []
    for i in range(nframe):
        fstart = i * fshift
        fend = fstart + flen

        frame = waveform[fstart:fend]
        frames.append(frame)

        label = labels.frame_mid_pitches(fstart, fend)
        hotvector = dataset_label.list2hotvector(label)
        hotvectors.append(hotvector)

    frames = np.array(frames, dtype=np.float32)
    hotvectors = np.array(hotvectors, dtype=np.float32)
    return frames, hotvectors
