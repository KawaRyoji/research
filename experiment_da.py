from machine_learning.model import hyper_params
from models.da_net.da_spec import da_spec
from models.da_net.da_logspec import da_logspec
from models.da_net.da_waveform import da_waveform

from machine_learning.k_fold_cross_validation import k_fold_cross_validation

params = hyper_params(batch_size=32, epochs=64, epoch_size=500, learning_rate=0.0001)

train_data = "./resource/musicnet16k/train_data"
train_labels = "./resource/musicnet16k/train_labels"
test_data = "./resource/musicnet16k/test_data"
test_labels = "./resource/musicnet16k/test_labels"
output_dir = "./experiment/da_logspec_mono"

# model = da_logspec.from_dir(train_data, train_labels, test_data, test_labels, output_dir)
model = da_logspec.from_solo_instrument(output_dir)

model.create_train_set(normalize=False)
model.create_test_set(normalize=False)
k = 5

model = k_fold_cross_validation(model, params, k)
model.train(
    monitor_best_cp="val_F1",
    monitor_mode="max",
    valid_size=params.batch_size * params.epoch_size // k,
)
model.test()

model.box_plot_history(metrics=["precision", "recall", "F1"])
model.plot_average_history()


def _kcv_predict(rawfile_name: str, model: k_fold_cross_validation, normalize=True):
    import pandas as pd
    import os

    test_res = pd.read_csv(os.path.join(model.result_dir, "test_res.csv"))
    res = pd.DataFrame.max(test_res, axis="index")
    model.predict_from_raw_data(
        "./resource/musicnet16k/test_data/" + rawfile_name + ".wav",
        "./resource/musicnet16k/test_labels/" + rawfile_name + ".csv",
        save_fig_path=os.path.join(
            model.result_dir, "figures/predict_" + rawfile_name + ".wav.png"
        ),
        model_weight_path=os.path.join(
            model.result_dir, "model_weights/cp_best_" + str(res[0]) + ".ckpt"
        ),
        normalize=normalize,
    )
    model.predict_from_raw_data(
        "./resource/musicnet16k/test_data/" + rawfile_name + ".wav",
        "./resource/musicnet16k/test_labels/" + rawfile_name + ".csv",
        save_fig_path=os.path.join(
            model.result_dir, "figures/predict_" + rawfile_name + ".wav_th0.5.png"
        ),
        model_weight_path=os.path.join(
            model.result_dir, "model_weights/cp_best_" + str(res[0]) + ".ckpt"
        ),
        normalize=normalize,
        threshold=0.5,
    )

_kcv_predict("2191", model, normalize=False)
