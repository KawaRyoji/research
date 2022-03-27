from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
import math
from audio.settings import params


class spectrogram:
    def __init__(self, spec, settings: params, ismel, isdB) -> None:
        self.spec = spec
        self.settings = settings
        self.ismel = ismel
        self.isdB = isdB

    @classmethod
    def make(cls, waveform, settings: params, dB=True, mel=True):
        if mel:
            spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=settings.fs,
                n_fft=settings.fftl,
                hop_length=settings.fshift,
                win_length=settings.flen,
            )
        else:
            spec = np.square(
                librosa.core.stft(
                    waveform,
                    n_fft=settings.fftl,
                    hop_length=settings.fshift,
                    win_length=settings.flen,
                )
            )

        if dB:
            spec = librosa.core.power_to_db(spec, ref=np.max)

        return cls(spec, settings, mel, dB)

    def show(self):
        librosa.display.specshow(
            self.spec, sr=self.settings.fs, x_axis="ms", y_axis="linear"
        )
        plt.colorbar()
        plt.show()

    def saveimg(self, filepath, gray_scale=False):
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.ioff()
        if gray_scale:
            plt.gray()
        else:
            plt.set_cmap("magma")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(filepath, np.flipud(self.spec))
        plt.close()

    def separate_images(self, overlap: int):
        assert overlap > 0 and overlap < 100
        width = len(self.spec)
        shift_len = math.floor(width * overlap / 100)
        nimg = math.floor(((len(self.spec[0]) - (width - shift_len)) / shift_len))

        for i in range(nimg):
            fstart = shift_len * i
            fend = fstart + width

            yield fstart * self.settings.fshift, fend * self.settings.fshift, spectrogram(
                self.spec[:, fstart:fend], self.settings, self.ismel, self.isdB
            )

    def __str__(self) -> str:
        return "shape:\t\t\t{}x{}\nsampling frequency:\t{}\tHz\nflame shift:\t\t{}\tpoint\nflame length:\t\t{}\tpoint\nfft length:\t\t{}\tpoint".format(
            len(self.spec),
            len(self.spec[0]),
            self.fs,
            self.fshift,
            self.flen,
            self.fftl,
        )
