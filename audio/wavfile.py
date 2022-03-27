import numpy as np
import librosa
import soundfile

class wavfile:
    def __init__(self, audiodata, fs) -> None:
        self.fs = fs
        self.data = np.array(audiodata, dtype=np.float32)

    @classmethod
    def read(cls, file_path, fs=None):
        wfile, fs = librosa.core.load(file_path, sr=fs)
        wav = cls(wfile, fs)
        return wav
    
    def resample(self, target_fs: int):
        if self.fs == target_fs:
            return self

        converted_waveform = librosa.core.resample(
            y=self.data, orig_sr=self.fs, target_sr=target_fs)

        return wavfile(converted_waveform, target_fs)

    def write(self, target_path: str, target_bits: int = 16):
        soundfile.write(target_path, self.data, self.fs,
                        'PCM_'+target_bits.__str__())

    def __str__(self) -> str:
        return "sampling frequency:{}\nlength of data:{}".format(self.fs, len(self.data))
