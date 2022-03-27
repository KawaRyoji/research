from machine_learning.interfaces import Imachine_learning
import numpy as np
import keras

audio_samples = 1024
shift_samples = 256

class crepe_mytask_waveform(Imachine_learning):
    def create_model(self) -> keras.Model:
        import models.crepe 
        return models.crepe.create_model()
    
    def _create_dataset_process(self, data_path: str, label_path: str) -> tuple:
        from audio.wavfile import wavfile
        from musicnet.annotation import dataset_label

        waveform = wavfile.read(data_path)
        labels = dataset_label.load(label_path)
        waveform = waveform.data
        
        nframe = 1 + int((len(waveform) - audio_samples) / shift_samples)    
        frames = []
        hotvectors = []
        for i in range(nframe):
            fstart = i*shift_samples
            fend = fstart + audio_samples
            
            frame = waveform[fstart:fend]
            frames.append(frame)
            
            label = labels.frame_mid_pitches(fstart, fend)
            hotvector = dataset_label.list2hotvector(label)
            hotvectors.append(hotvector)
        
        frames = np.array(frames, dtype=np.float32)
        hotvectors = np.array(hotvectors, dtype=np.float32)
        return frames, hotvectors
