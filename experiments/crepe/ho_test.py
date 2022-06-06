from experiments.ho_experiment import ho_experiment
from machine_learning.dataset import dataset
from machine_learning.parameter import hyper_params
from models.CREPE import CREPE
from experiments.features.waveform import construct_process
from musicnet.musicnet import solo_instrument

(
    train_data_paths,
    train_label_paths,
    test_data_paths,
    test_label_paths,
) = solo_instrument()

experimental_result_dir = "./experimental_results/ho_test"
params = hyper_params(32, 16, epoch_size=500, learning_rate=0.0001)

train_set = dataset(train_data_paths, train_label_paths, construct_process)
test_set = dataset(train_data_paths, train_label_paths, construct_process)

model = CREPE()
ex = ho_experiment(
    model=model,
    train_set=train_set,
    test_set=test_set,
    params=params,
    experimental_result_dir=experimental_result_dir,
)

ex.prepare_dataset()
ex.train()
ex.test()
