import os
from typing import List

import keras
import keras.backend as K
import numpy as np
import pandas as pd
from machine_learning.dataset import data_sequence, dataset
from machine_learning.holdout.holdout_result import holdout_result
from machine_learning.metrics import F1_from_log
from machine_learning.model import learning_model
from machine_learning.parameter import hyper_params


class holdout:
    def __init__(
        self, model: learning_model, params: hyper_params, results: holdout_result
    ) -> None:
        self.model = model
        self.params = params
        self.result = results

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        valid_split: float,
        monitor_best_cp="val_loss",
        monitor_mode="auto",
        callbacks: List[keras.callbacks.Callback] = None,
    ) -> None:
        model = self.model.create_model()

        if not model._is_compiled:
            print("Model should be compiled.")
            return

        K.set_value(model.optimizer.lr, self.params.learning_rate)

        train_x, train_y, valid_x, valid_y = dataset.split_data(x, y, valid_split)
        self.params.save_to_json(os.path.join(self.result.results_dir, "params.json"))

        if callbacks is None:
            callbacks = []

        cp_callback = keras.callbacks.ModelCheckpoint(
            self.result.model_weight_path,
            monitor=monitor_best_cp,
            mode=monitor_mode,
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
        )
        lg_callback = keras.callbacks.CSVLogger(self.result.history_path)
        callbacks.append(cp_callback)
        callbacks.append(lg_callback)

        train_sequence = data_sequence(
            train_x,
            train_y,
            self.params.batch_size,
            batches_per_epoch=self.params.epoch_size,
        )

        del train_x
        del train_y

        model.fit(
            x=train_sequence,
            epochs=self.params.epochs,
            callbacks=callbacks,
            validation_data=(valid_x, valid_y),
        )

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ):
        model = self.model.create_model()
        model.load_weights(self.result.model_weight_path)

        result = model.evaluate(x, y, verbose=2)
        df = pd.DataFrame(result, columns=model.metrics_names)

        df = F1_from_log(df)
        df.to_csv(os.path.join(self.result.results_dir, "test_res.csv"))

    def predict(
        self,
        x: np.ndarray,
        model_weight_path: str,
    ):
        model = self.model.create_model()
        model.load_weights(model_weight_path)

        prediction = model.predict(x)

        return prediction
