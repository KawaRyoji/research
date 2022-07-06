import numpy as np
from machine_learning.dataset import dataset
from experiments.features.spectrogram import construct_process
from scipy.stats import zscore


data_list = []


data, label = construct_process(
    "./resource/musicnet16k/train_data/1727.wav",
    "./resource/musicnet16k/train_labels/1727.csv",)


print(data.shape)

data_list.extend(data)

data_list = np.array(data_list)

print(data_list.shape)

