import keras
from machine_learning.interfaces import Imachine_learning


class crepe_mytask_logspec(Imachine_learning):
    def create_model(self) -> keras.Model:
        import models.crepe
        return models.crepe.create_model()

    def _create_dataset_process(self, data_path: str, label_path: str) -> tuple:
        from scipy.signal.windows.windows import hann
        from scipy.fft import fft
        import numpy as np
        from audio.wavfile import wavfile
        from musicnet.annotation import dataset_label

        fft_len = 2048  # 分解能 8Hz
        flen = 1024  # 時間長 64ms
        fshift = 256  # 時間長 16ms
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
            frame = 20 * np.log10(frame, out=np.zeros_like(frame), where=frame!=0)
            frame = frame[: fft_len // 2]

            label = labels.frame_mid_pitches(fstart-pad, fend-pad)
            label = dataset_label.list2hotvector(label)

            frames.append(frame)
            hotvectors.append(label)

        frames = np.array(frames, dtype=np.float32)
        hotvectors = np.array(hotvectors, dtype=np.float32)
        return frames, hotvectors
