from pathlib import Path
from audio.wavfile import wavfile
import os

from musicnet.annotation import dataset_label
from musicnet import musicnet

target_fs = 16000

train_data_dir_path = "./resource/musicnet/train_data_origin"
train_labels_dir_path = "./resource/musicnet/train_labels_origin"
test_data_dir_path = "./resource/musicnet/test_data_origin"
test_labels_dir_path = "./resource/musicnet/test_labels_origin"

output_train_data_dir_path = "./resource/musicnet16k/train_data"
output_train_labels_dir_path = "./resource/musicnet16k/train_labels"
output_test_data_dir_path = "./resource/musicnet16k/test_data"
output_test_labels_dir_path = "./resource/musicnet16k/test_labels"

Path.mkdir(Path(output_train_data_dir_path), parents=True, exist_ok=True)
Path.mkdir(Path(output_train_labels_dir_path), parents=True, exist_ok=True)
Path.mkdir(Path(output_test_data_dir_path), parents=True, exist_ok=True)
Path.mkdir(Path(output_test_labels_dir_path), parents=True, exist_ok=True)

for data_filename in os.listdir(train_data_dir_path):
    wav = wavfile.read(os.path.join(train_data_dir_path, data_filename))
    wav = wav.resample(target_fs)
    wav.write(os.path.join(output_train_data_dir_path, data_filename))

    label_filename = os.path.splitext(data_filename)[0] + ".csv"
    label = dataset_label.load(os.path.join(train_labels_dir_path, label_filename))
    label.convert_fs(musicnet.recording_sample_rate, target_fs)
    label.save(os.path.join(output_train_labels_dir_path, label_filename))

for data_filename in os.listdir(test_data_dir_path):
    wav = wavfile.read(os.path.join(test_data_dir_path, data_filename))
    wav = wav.resample(target_fs)
    wav.write(os.path.join(output_test_data_dir_path, data_filename))

    label_filename = os.path.splitext(data_filename)[0] + ".csv"
    label = dataset_label.load(os.path.join(test_labels_dir_path, label_filename))
    label.convert_fs(musicnet.recording_sample_rate, target_fs)
    label.save(os.path.join(output_test_labels_dir_path, label_filename))
