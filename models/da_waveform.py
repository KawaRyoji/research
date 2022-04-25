import keras
from machine_learning.interfaces import Imachine_learning
import numpy as np
from musicnet.musicnet import solo_instrumental_train, solo_instrumental_test

flen = 1024
fshift = 256

class da_waveform(Imachine_learning):
    def create_model(self) -> keras.Model:
        import models.DA_Net
        return models.DA_Net.create_model()

    def _create_dataset_process(self, data_path: str, label_path: str) -> tuple:
        from audio.wavfile import wavfile
        from musicnet.annotation import dataset_label

        waveform = wavfile.read(data_path)
        labels = dataset_label.load(label_path)
        waveform = waveform.data
        
        nframe = 1 + (len(waveform) - flen) // fshift    
        frames = []
        hotvectors = []
        for i in range(nframe):
            fstart = i*fshift
            fend = fstart + flen
            
            frame = waveform[fstart:fend]
            frames.append(frame)
            
            label = labels.frame_mid_pitches(fstart, fend)
            hotvector = dataset_label.list2hotvector(label)
            hotvectors.append(hotvector)
        
        frames = np.array(frames, dtype=np.float32)
        hotvectors = np.array(hotvectors, dtype=np.float32)
        return frames, hotvectors
    
    
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