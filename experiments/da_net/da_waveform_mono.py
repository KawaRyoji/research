from experiments.kcv_experiment import kcv_experiment
from machine_learning.dataset import dataset
from machine_learning.parameter import hyper_params
from models.DA_Net import DA_Net
from experiments.features.waveform import construct_process
from musicnet.musicnet import solo_instrument

(
    train_data_paths,
    train_label_paths,
    test_data_paths,
    test_label_paths,
) = solo_instrument()

experimental_result_dir = "./experimental_results/da_waveform_mono"
params = hyper_params(32, 64, epoch_size=500, learning_rate=0.0001)

train_set = dataset(train_data_paths, train_label_paths, construct_process)
test_set = dataset(train_data_paths, train_label_paths, construct_process)

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
