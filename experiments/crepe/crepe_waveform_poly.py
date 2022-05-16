from experiments.kcv_experiment import kcv_experiment
from machine_learning.dataset import dataset
from machine_learning.parameter import hyper_params
from models.CREPE import CREPE
from experiments.features.waveform import construct_process

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


experimental_result_dir = "./experimental_results/crepe_waveform_poly"
params = hyper_params(32, 32, epoch_size=500, learning_rate=0.0001)

model = CREPE()
ex = kcv_experiment(
    model=model,
    train_set=train_set,
    test_set=test_set,
    params=params,
    k=5,
    experimantal_result_dir=experimental_result_dir,
)

ex.prepare_dataset()
ex.train()
ex.test()
