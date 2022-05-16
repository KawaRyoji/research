import os
import keras.backend as K
import keras
import numpy as np
import pandas as pd
from machine_learning.model import learning_model
from machine_learning.kcv.kcv_result import kcv_result
from machine_learning.learning_history import learning_history
from machine_learning.metrics import F1_from_log
from machine_learning.parameter import hyper_params
import machine_learning.plot as plot
from machine_learning.dataset import data_sequence


class kcv:
    """
    k分割交差検証により学習させるインタフェースです

    Imachine_learningに依存しています
    """

    def __init__(
        self, model: learning_model, params: hyper_params, k: int, results: kcv_result
    ) -> None:
        self.model = model
        self.params = params
        self.k = k
        self.result = results

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        monitor_best_cp="val_loss",
        monitor_mode="auto",
        callbacks=None,
        valid_size=None,
    ):
        self.params.save_to_json(os.path.join(self.result.results_dir, "params.json"))

        fold_size = len(x) // self.k
        for fold in range(self.k):
            model = self.model.create_model()

            if not model._is_compiled:
                print("Model should be compiled.")
                return

            K.set_value(model.optimizer.lr, self.params.learning_rate)

            if callbacks is None:
                callbacks_f = []
            else:
                callbacks_f = callbacks

            cp_callback = keras.callbacks.ModelCheckpoint(
                os.path.join(
                    self.result.model_weights_dir,
                    "cp_best_fold{}.ckpt".format(fold),
                ),
                monitor=monitor_best_cp,
                mode=monitor_mode,
                save_weights_only=True,
                save_best_only=True,
                verbose=1,
            )
            lg_callback = keras.callbacks.CSVLogger(
                os.path.join(
                    self.result.histories_dir, "history_fold{}.csv".format(fold)
                )
            )
            callbacks_f.append(cp_callback)
            callbacks_f.append(lg_callback)

            split_s = fold * fold_size
            split_e = split_s + fold_size
            mask = np.ones(len(x), dtype=bool)
            mask[split_s:split_e] = False
            x_valid = x[split_s:split_e]
            y_valid = y[split_s:split_e]
            # NOTE: メモリをバカ食いする
            x_train = x[mask]
            y_train = y[mask]

            del mask
            sequence = data_sequence(
                x_train,
                y_train,
                self.params.batch_size,
                batches_per_epoch=self.params.epoch_size,
            )

            if valid_size is not None:
                x_valid = x_valid[:valid_size]
                y_valid = y_valid[:valid_size]

            model.fit(
                x=sequence,
                epochs=self.params.epochs,
                callbacks=callbacks_f,
                validation_data=(x_valid, y_valid),
            )

    def test(self, x: np.ndarray, y: np.ndarray):
        results = []
        for fold in range(self.k):
            model = self.model.create_model()
            model.load_weights(
                os.path.join(
                    self.result.model_weights_dir,
                    "cp_best_fold{}.ckpt".format(fold),
                )
            )
            res = model.evaluate(x, y, verbose=2)
            results.append(res)

        df = pd.DataFrame(
            results,
            index=["fold{}".format(i) for i in range(self.k)],
            columns=model.metrics_names,
        )

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

    def box_plot_history(self, metrics: list = None):
        """
        テスト結果を箱ひげ図にてプロットします

        ## Params:
            metrics (list, optional): プロットする評価値を指定できます
        """
        plot.box_plot_history(
            os.path.join(
                self.result.figures_dir,
                "test_res.png",
            ),
            os.path.join(self.result.results_dir, "test_res.csv"),
            metrics=metrics,
        )

    def plot_average_history(self):
        histories = learning_history.from_dir(self.result.histories_dir)
        average = learning_history.average(histories)

        plot.plot_history(average, self.result.average_result_dir)
