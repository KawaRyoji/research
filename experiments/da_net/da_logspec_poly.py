from experiments.kcv_experiment import kcv_experiment
from machine_learning.dataset import dataset
from machine_learning.parameter import hyper_params
from models.DA_Net import DA_Net
from experiments.features.log_spectrum import construct_process
from musicnet.musicnet import solo_instrument

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

experimental_result_dir = "./experimental_results/da_logspec_poly"
params = hyper_params(32, 64, epoch_size=500, learning_rate=0.0001)

model = DA_Net()
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
