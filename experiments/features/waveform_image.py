from audio.wavfile import wavfile
from musicnet.annotation import dataset_label
import numpy as np


def construct_process(
    data_path: str,
    label_path: str,
    flen=1024,
    fshift=256,
    time_len=16,
) -> tuple:
    waveform = wavfile.read(data_path)
    labels = dataset_label.load(label_path)
    waveform = waveform.data

    nframe = 1 + (len(waveform) - flen) // fshift
    nimage = nframe // time_len
    frames = [[] for _ in range(nimage)]
    hotvectors = [[] for _ in range(nimage)]

    for i in range(nimage):
        for j in range(time_len):
            fstart = (i * time_len + j) * fshift
            fend = fstart + flen

            frame = waveform[fstart:fend]
            label = labels.frame_mid_pitches(fstart, fend)
            hotvector = dataset_label.list2hotvector(label)

            frames[i].append(frame)
            hotvectors[i].append(hotvector)

    frames = np.array(frames, dtype=np.float32)
    hotvectors = np.array(hotvectors, dtype=np.float32)
    return frames, hotvectors
