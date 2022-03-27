import keras
import numpy as np
from audio.wavfile import wavfile
from machine_learning.interfaces import Imachine_learning
from musicnet.annotation import dataset_label
from models.crepe_origintask_waveform import (
    solo_instrumental_train,
    solo_instrumental_test,
)

flen = 1024  # 時間長 64ms  周波数分解能 15.625 Hz
fshift = 256  # 時間長 16ms
fs = 16000


class crepe_origintask_spec_not_windowed(Imachine_learning):
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
        from scipy.signal.windows.windows import hann
        from scipy.fft import fft

        fft_len = 2048  # 分解能 8Hz

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
