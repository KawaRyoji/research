import os
from machine_learning.k_fold_cross_validation import k_fold_cross_validation
from machine_learning.model import hyper_params
import pandas as pd
from models.crepe_origintask_waveform import crepe_origintask_waveform
from models.crepe_origintask_spec import crepe_origintask_spec
from models.crepe_mytask_waveform import crepe_mytask_waveform
from models.crepe_mytask_spec import crepe_mytask_spec
from models.crepe_mytask_mel_bins128 import crepe_mytask_mel_bins128
from models.crepe_mytask_mel_bins256 import crepe_mytask_mel_bins256
from models.crepe_mytask_logspec import crepe_mytask_logspec
from models.crepe_origintask_spec_not_windowed import crepe_origintask_spec_not_windowed
from models.crepe_origintask_waveform_windowed import crepe_origintask_waveform_windowed

params = hyper_params(32, 32, epoch_size=500, learning_rate=0.0002)
k = 5

train_data = "./resource/musicnet16k/train_data"
train_labels = "./resource/musicnet16k/train_labels"
test_data = "./resource/musicnet16k/test_data"
test_labels = "./resource/musicnet16k/test_labels"

origintask_spec = crepe_origintask_spec.from_solo_instrument(
    "./experiment/crepe_origintask_spec",
)

origintask_waveform = crepe_origintask_waveform.from_solo_instrument(
    "./experiment/crepe_origintask_waveform",
)

mytask_waveform = crepe_mytask_waveform.from_dir(
    train_data,
    train_labels,
    test_data,
    test_labels,
    "./experiment/crepe_mytask_waveform",
)

mytask_spec = crepe_mytask_spec.from_dir(
    train_data,
    train_labels,
    test_data,
    test_labels,
    "./experiment/crepe_mytask_spec",
)

mytask_logspec = crepe_mytask_logspec.from_dir(
    train_data,
    train_labels,
    test_data,
    test_labels,
    "./experiment/crepe_mytask_logspec",
)

mytask_mel_bins128 = crepe_mytask_mel_bins128.from_dir(
    train_data,
    train_labels,
    test_data,
    test_labels,
    "./experiment/crepe_mytask_mel_bins128",
)
mytask_mel_bins256 = crepe_mytask_mel_bins256.from_dir(
    train_data,
    train_labels,
    test_data,
    test_labels,
    "./experiment/crepe_mytask_mel_bins256",
)

origintask_spec_not_windowed = crepe_origintask_spec_not_windowed.from_solo_instrument(
    "./experiment/crepe_origintask_spec_not_windowed",
)

origintask_waveform_windowed = crepe_origintask_waveform_windowed.from_solo_instrument(
    "./experiment/crepe_origintask_waveform_windowed"
)


models = [
    origintask_spec,
    origintask_waveform,
    mytask_waveform,
    mytask_spec,
    mytask_logspec,
    mytask_mel_bins128,
    mytask_mel_bins256,
    origintask_spec_not_windowed,
]


def _run():
    for m in models:
        # _kcv(m)
        model = k_fold_cross_validation(m, params, k)
        _plot_avg_history(model)
        _box_plot_history(model)
        _kcv_predict("2191", model)
    _plot_avg_compare()
    _box_plot_compare()


def _exp():
    output_dir = "./experiment/crepe_origintask_waveform_windowed"
    ml = crepe_origintask_waveform_windowed.from_solo_instrument(output_dir)

    ml.create_train_set()
    ml.create_test_set()
    _kcv(ml)
    _plot_avg_history(output_dir)
    _box_plot_history(output_dir)


def _kcv(model: k_fold_cross_validation):
    model.train(
        monitor_best_cp="val_F1",
        monitor_mode="max",
        valid_size=params.batch_size * params.epoch_size // k,
    )
    model.test()


def _box_plot_compare():
    k_fold_cross_validation.box_plot_history_compare(
        "./output/compare_mytask/kcv.png",
        mytask_waveform,
        mytask_spec,
        mytask_mel_bins256,
        legend=["waveform", "spectrum", "mel spectrum"],
        metrics=["precision", "recall", "F1"],
    )

    k_fold_cross_validation.box_plot_history_compare(
        "./output/compare_mytask_mel/kcv.png",
        mytask_mel_bins128,
        mytask_mel_bins256,
        legend=["128 bins", "256 bins"],
        metrics=["precision", "recall", "F1"],
    )

    k_fold_cross_validation.box_plot_history_compare(
        "./output/compare_origintask/kcv.png",
        origintask_waveform,
        origintask_spec,
        legend=["waveform", "spectrum"],
        metrics=["precision", "recall", "F1"],
    )

    k_fold_cross_validation.box_plot_history_compare(
        "./output/compare_mytask_spec/kcv.png",
        mytask_spec,
        mytask_logspec,
        mytask_mel_bins256,
        legend=["spectrum", "log spectrum", "mel spectrum"],
        metrics=["precision", "recall", "F1"],
    )

    k_fold_cross_validation.box_plot_history_compare(
        "./output/compare_windowed_spec/kcv.png",
        origintask_spec,
        origintask_spec_not_windowed,
        legend=["windowed", "not windowed"],
        metrics=["precision", "recall", "F1"],
    )

    k_fold_cross_validation.box_plot_history_compare(
        "./output/compare_windowed_waveform/kcv.png",
        origintask_waveform,
        origintask_waveform_windowed,
        legend=["not windowed", "windowed"],
        metrics=["precision", "recall", "F1"],
    )


def _plot_avg_compare():
    k_fold_cross_validation.plot_compare_average_history(
        "./output/compare_mytask_spec",
        mytask_spec,
        mytask_logspec,
        legend=["linear", "logarithm"],
        metrics=["val_precision", "val_recall", "val_F1"],
    )

    k_fold_cross_validation.plot_compare_average_history(
        "./output/compare_mytask",
        mytask_waveform,
        mytask_spec,
        mytask_mel_bins256,
        legend=["waveform", "spectrum", "mel spectrum"],
        metrics=["val_precision", "val_recall", "val_F1"],
    )

    k_fold_cross_validation.plot_compare_average_history(
        "./output/compare_mytask_mel",
        mytask_mel_bins128,
        mytask_mel_bins256,
        legend=["128 bins", "256 bins"],
        metrics=["val_precision", "val_recall", "val_F1"],
    )

    k_fold_cross_validation.plot_compare_average_history(
        "./output/compare_origintask",
        origintask_waveform,
        origintask_spec,
        legend=["waveform", "spectrum"],
        metrics=["val_precision", "val_recall", "val_F1"],
    )

    k_fold_cross_validation.plot_compare_average_history(
        "./output/compare_windowed_spec",
        origintask_spec,
        origintask_spec_not_windowed,
        legend=["windowed", "not windowed"],
        metrics=["val_precision", "val_recall", "val_F1"],
    )

    k_fold_cross_validation.plot_compare_average_history(
        "./output/compare_windowed_waveform",
        origintask_waveform,
        origintask_waveform_windowed,
        legend=["not windowed", "windowed"],
        metrics=["val_precision", "val_recall", "val_F1"],
    )


def _box_plot_history(model: k_fold_cross_validation):
    model.box_plot_history(metrics=["precision", "recall", "F1"])


def _plot_avg_history(model: k_fold_cross_validation):
    model.plot_average_history()


def _kcv_predict(rawfile_name: str, model: k_fold_cross_validation):
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
        threshold=0.5,
    )


# _run()

model = k_fold_cross_validation(origintask_spec_not_windowed, params, k)
model.train(
    monitor_best_cp="val_F1",
    monitor_mode="max",
    valid_size=params.batch_size * params.epoch_size // k,
)
model.test()
model.box_plot_history(metrics=["precision", "recall", "F1"])
model.plot_average_history()