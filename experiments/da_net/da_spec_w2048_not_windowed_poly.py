import tensorflow as tf

from experiments.kcv_experiment import kcv_experiment
from machine_learning.dataset import dataset
from machine_learning.parameter import hyper_params
from models.DA_Net import DA_Net
from experiments.features.spectrum_not_windowed import construct_process

phisical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(phisical_devices[0], "GPU")
tf.config.experimental.set_memory_growth(phisical_devices[0], True)

train_data_dir = "./resource/musicnet16k/train_data"
train_label_dir = "./resource/musicnet16k/train_labels"
test_data_dir = "./resource/musicnet16k/test_data"
test_label_dir = "./resource/musicnet16k/test_labels"


train_set = dataset.from_dir(
    train_data_dir,
    train_label_dir,
    construct_process,
)

test_set = dataset.from_dir(
    test_data_dir,
    test_label_dir,
    construct_process,
)

experimental_result_dir = "./experimental_results/da_spec_w2048_not_windowed_poly"
params = hyper_params(32, 64, epoch_size=500, learning_rate=0.0001)

model = DA_Net()
ex = kcv_experiment(
    model=model,
    train_set=train_set,
    test_set=test_set,
    params=params,
    k=5,
    experimental_result_dir=experimental_result_dir,
)

flen = 2048

ex.prepare_dataset(normalize=False, flen=flen)
ex.train()
ex.test()
ex.plot_prediction(
    "./resource/musicnet16k/test_data/2556.wav",
    "./resource/musicnet16k/test_labels/2556.csv",
    threshold=0.5,
    flen=flen,
)
ex.plot_prediction(
    "./resource/musicnet16k/test_data/2628.wav",
    "./resource/musicnet16k/test_labels/2628.csv",
    threshold=0.5,
    flen=flen,
)