import os
from typing import List

import machine_learning.plot as plot
import numpy as np
from tensorflow.keras.callbacks import Callback
from machine_learning.dataset import dataset
from machine_learning.holdout.holdout import holdout
from machine_learning.holdout.holdout_result import holdout_result
from machine_learning.learning_history import learning_history
from machine_learning.model import learning_model
from machine_learning.parameter import hyper_params


class ho_experiment:
    def __init__(
        self,
        model: learning_model,
        train_set: dataset,
        test_set: dataset,
        params: hyper_params,
        experimental_result_dir: str,
    ) -> None:
        self.results = holdout_result(experimental_result_dir)
        self.model = holdout(model, params, self.results)
        self.train_set = train_set
        self.test_set = test_set
        self.params = params

    def prepare_dataset(self, normalize=False, **kwargs):
        self.train_set.construct(
            os.path.join(self.results.root_dir, "train"), normalize=normalize, **kwargs
        )
        self.test_set.construct(
            os.path.join(self.results.root_dir, "test"), normalize=normalize, **kwargs
        )

    def train(
        self,
        valid_split=0.25,
        callbacks: List[Callback] = None,
        valid_limit: int = None,
    ):
        print("loading dataset ...")
        x, y = self.train_set.load(os.path.join(self.results.root_dir, "train"))
        print("loading dataset is done")
        print("train data shape:", x.shape)
        print("train labels shape:", y.shape)
        print(self.params)

        self.model.train(
            x,
            y,
            valid_split,
            monitor_best_cp="val_F1",
            monitor_mode="max",
            callbacks=callbacks,
            valid_limit=valid_limit,
        )

        self.plot_result()

    def test(self):
        x, y = self.test_set.load(
            os.path.join(self.results.root_dir, "test"), shuffle=False
        )

        self.model.test(x, y)

    def plot_prediction(
        self,
        data_path: str,
        label_path: str,
        normalize=False,
        threshold: float = None,
        **kwargs
    ):
        x, y = self.train_set._construct_process(data_path, label_path, **kwargs)
        basename = os.path.basename(data_path)

        if normalize:
            x = dataset.normalize_data(x)

        prediction = self.model.predict(x, self.results.model_weight_path)

        plot.plot_activation_with_labels(
            y,
            prediction,
            os.path.join(
                self.results.figures_dir,
                "predict_" + basename + ".png",
            ),
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

    def plot_result(self):
        history = learning_history.from_path(self.results.history_path)
        plot.plot_history(history, self.results.figures_dir)
