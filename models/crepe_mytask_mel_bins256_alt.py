import keras
from machine_learning.interfaces import Imachine_learning


class crepe_mytask_mel_bins256_alt(Imachine_learning):
    def create_model(self) -> keras.Model:
        from keras.layers import Input, Reshape, Conv2D, BatchNormalization
        from keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense
        from keras.models import Model
        from keras.metrics import Precision, BinaryAccuracy, Recall
        from machine_learning.metrics import F1

        input_size = 256
        capacity_multiplier = 32
        layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        widths = [
            input_size // 2,
            input_size // 16,
            input_size // 16,
            input_size // 16,
            input_size // 16,
            input_size // 16,
        ]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        x = Input(shape=(input_size,), name="input", dtype="float32")
        y = Reshape(target_shape=(input_size, 1, 1), name="input-reshape")(x)

        for l, f, w, s in zip(layers, filters, widths, strides):
            y = Conv2D(
                f, (w, 1), strides=s, padding="same", activation="relu", name="conv%d" % l
            )(y)
            y = BatchNormalization(name="conv%d-BN" % l)(y)
            y = MaxPool2D(
                pool_size=(min(y.shape[1], 2), 1), strides=None, padding="valid", name="conv%d-maxpool" % l
            )(y)
            y = Dropout(0.25, name="conv%d-dropout" % l)(y)

        y = Permute((2, 1, 3), name="transpose")(y)
        y = Flatten(name="flatten")(y)
        y = Dense(128, activation="sigmoid", name="classifier")(y)

        model = Model(inputs=x, outputs=y)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="binary_crossentropy",
            metrics=[
                BinaryAccuracy(),
                F1,
                Precision(name="precision"),
                Recall(name="recall"),
            ],
        )
        return model

    def _create_dataset_process(self, data_path: str, label_path: str) -> tuple:
        from scipy.signal.windows.windows import hann
        from scipy.fft import fft
        import numpy as np
        from audio.wavfile import wavfile
        from musicnet.annotation import dataset_label
        from librosa.filters import mel
        from models.crepe import sampling_freq
        from util.calc import square
        
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
