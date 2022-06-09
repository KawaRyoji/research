import os

import machine_learning.plot as plot
import numpy as np
import pandas as pd
from machine_learning.dataset import dataset
from machine_learning.kcv.kcv import kcv
from machine_learning.kcv.kcv_result import kcv_result
from machine_learning.learning_history import learning_history
from machine_learning.model import learning_model
from machine_learning.parameter import hyper_params


class kcv_experiment:
    """
    k分割交差検証の実験用クラスです。
    自身の実験に即して書かれているので用いる場合は注意してください。
    """

    def __init__(
        self,
        model: learning_model,
        train_set: dataset,
        test_set: dataset,
        params: hyper_params,
        k: int,
        experimantal_result_dir: str,
    ) -> None:
        self.results = kcv_result(experimantal_result_dir, k)
        self.model = kcv(model, params, k, self.results)
        self.train_set = train_set
        self.test_set = test_set
        self.params = params
        self.k = k

    def prepare_dataset(self, normalize=False, **kwargs):
        self.train_set.construct(
            os.path.join(self.results.root_dir, "train"), normalize=normalize, **kwargs
        )
        self.test_set.construct(
            os.path.join(self.results.root_dir, "test"), normalize=normalize, **kwargs
        )

    def train(self):
        print("loading dataset ...")
        x, y = self.train_set.load(os.path.join(self.results.root_dir, "train"))
        print("loading dataset is done")
        print("train data shape:", x.shape)
        print("train labels shape:", y.shape)
        print(self.params)

        self.model.train(
            x,
            y,
            monitor_best_cp="val_F1",
            monitor_mode="max",
            valid_size=self.params.batch_size * self.params.epoch_size // self.k,
        )

        self.plot_fold_result()
        self.plot_fold_average()

    def plot_fold_result(self):
        histories = learning_history.from_dir(self.results.histories_dir)
        for fold, history in enumerate(histories):
            plot.plot_history(
                history,
                os.path.join(self.results.figures_dir, "figure_fold{}".format(fold)),
            )

    def plot_fold_average(self):
        histories = learning_history.from_dir(self.results.histories_dir)
        history = learning_history.average(*histories)
        plot.plot_history(history, self.results.average_result_dir)

    def test(self):
        x, y = self.test_set.load(
            os.path.join(self.results.root_dir, "test"), shuffle=False
        )

        self.model.test(x, y)

        history = learning_history.from_path(
            os.path.join(self.results.results_dir, "test_res.csv")
        )
        plot.box_plot_history(
            history,
            os.path.join(self.results.figures_dir, "test_res.png"),
            metrics=["precision", "recall", "F1"],
        )

    def plot_prediction(
        self, data_path: str, label_path: str, normalize=False, threshold: float = None
    ):
        test_res = pd.read_csv(os.path.join(self.results.results_dir, "test_res.csv"))
        res = pd.DataFrame.max(test_res, axis="index")
        model_weight_path = os.path.join(
            self.results.model_weights_dir, "cp_best_" + str(res[0]) + ".ckpt"
        )
        basename = os.path.basename(data_path)

        x, y = self.train_set._construct_process(data_path, label_path)
        if normalize:
            x = dataset.normalize_data(x)

        prediction = self.model.predict(x, model_weight_path)

        plot.plot_activation_with_labels(
            y,
            prediction,
            os.path.join(self.results.figures_dir, "predict_" + basename + ".png"),
        )

        if threshold is None:
            return

        prediction = np.where(prediction < threshold, 0, 1)

        plot.plot_activation_with_labels(
            y,
            prediction,
            os.path.join(
                self.results.figures_dir,
                "predict_" + basename + "_th{:.2f}.png".format(threshold),
            ),
        )
