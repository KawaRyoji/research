from experiments.ho_experiment import ho_experiment
from machine_learning.dataset import dataset
from machine_learning.parameter import hyper_params
from models.DA_Net import DA_Net
from experiments.features.waveform import construct_process
from keras import callbacks

train_data_dir = "./resource/musicnet16k/train_data"
train_label_dir = "./resource/musicnet16k/train_labels"
test_data_dir = "./resource/musicnet16k/test_data"
test_label_dir = "./resource/musicnet16k/test_labels"
experimental_result_dir = "./experimental_results/da_waveform_nolimit_poly"

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

params = hyper_params(32, 10000, epoch_size=500, learning_rate=0.0001)

model = DA_Net()
ex = ho_experiment(
    model=model,
    train_set=train_set,
    test_set=test_set,
    params=params,
    experimental_result_dir=experimental_result_dir,
)
es_callback = callbacks.EarlyStopping(patience=5)

ex.prepare_dataset(normalize=False)
ex.train(
    callbacks=[es_callback], valid_limit=params.batch_size * params.epoch_size // 4
)
ex.test()
ex.plot_prediction(
    "./resource/musicnet16k/test_data/2556.wav",
    "./resource/musicnet16k/test_labels/2556.csv",
    threshold=0.5,
)
ex.plot_prediction(
    "./resource/musicnet16k/test_data/2628.wav",
    "./resource/musicnet16k/test_labels/2628.csv",
    threshold=0.5,
)
