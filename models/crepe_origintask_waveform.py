import os
from audio.wavfile import wavfile
from machine_learning.interfaces import Imachine_learning
from musicnet.annotation import dataset_label
import keras
import numpy as np
from util.fig import graph_settings

flen = 1024  # 時間長 64ms  周波数分解能 15.625 Hz
fshift = 256  # 時間長 8ms
fs = 16000

# ソロ楽器の楽曲
solo_instrumental_train = [
    2186,
    2241,
    2242,
    2243,
    2244,
    2288,
    2289,
    2659,
    2217,
    2218,
    2219,
    2220,
    2221,
    2222,
    2293,
    2294,
    2295,
    2296,
    2297,
    2202,
    2203,
    2204,
]

solo_instrumental_test = [2191, 2298]


class crepe_origintask_waveform(Imachine_learning):
    @classmethod
    def from_solo_instrument(cls, output_dir):
        train_data_paths = list(
            map(
                lambda x: "./resource/musicnet16k/train_data/{}.wav".format(x),
                solo_instrumental_train,
            )
        )
        train_labels_paths = list(
            map(
                lambda x: "./resource/musicnet16k/train_labels/{}.csv".format(x),
                solo_instrumental_train,
            )
        )
        test_data_paths = list(
            map(
                lambda x: "./resource/musicnet16k/test_data/{}.wav".format(x),
                solo_instrumental_test,
            )
        )
        test_labels_paths = list(
            map(
                lambda x: "./resource/musicnet16k/test_labels/{}.csv".format(x),
                solo_instrumental_test,
            )
        )

        return cls(
            train_data_paths,
            train_labels_paths,
            test_data_paths,
            test_labels_paths,
            output_dir,
        )

    def create_model(self) -> keras.Model:
        import models.crepe
        return models.crepe.create_model()

    def _create_dataset_process(self, data_path: str, label_path: str) -> tuple:
        waveform = wavfile.read(data_path)
        labels = dataset_label.load(label_path)
        waveform = waveform.data

        nframe = 1 + int((len(waveform) - flen) / fshift)
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

            frame = waveform[fstart:fend]
            frames.append(frame)
            hotvectors.append(hotvector)

        frames = np.array(frames, dtype=np.float32)
        hotvectors = np.array(hotvectors, dtype=np.float32)
        return frames, hotvectors