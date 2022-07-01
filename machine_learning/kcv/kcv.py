import os
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from machine_learning.kcv.kcv_data_sequence import kcv_data_sequence
from machine_learning.model import learning_model
from machine_learning.kcv.kcv_result import kcv_result
from machine_learning.learning_history import learning_history
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from machine_learning.metrics import F1_from_log
from machine_learning.parameter import hyper_params
import machine_learning.plot as plot
from machine_learning.dataset import data_sequence
from typing import List


class kcv:
    """
    k分割交差検証を扱うクラスです。
    """

    def __init__(
        self, model: learning_model, params: hyper_params, k: int, results: kcv_result
    ) -> None:
        """
        Args:
            model (learning_model): 学習させるモデル
            params (hyper_params): 学習の際のパラメータ
            k (int): 分割する個数
            results (kcv_result): 結果を保存するディレクトリ
        """
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
        callbacks: List[Callback] = None,
        valid_size: int = None,
    ):
        """
        k分割交差検証で学習と検証を行います。

        Args:
            x (np.ndarray): 入力データ
            y (np.ndarray): 入力データに対応するラベル
            monitor_best_cp (str, optional): モデルの重みを保存するときにモニタリングする評価値
            monitor_mode (str, optional): モニタリングのモード
            callbacks (List[keras.callbacks.Callback], optional): コールバック関数を指定します
            valid_size (int, optional): 検証用データのサイズを制限します
        """
        self.params.save_to_json(os.path.join(self.result.results_dir, "params.json"))

        sequence = kcv_data_sequence(
            x,
            y,
            self.k,
            self.params.batch_size,
            batches_per_epoch=self.params.epoch_size,
            valid_size=valid_size,
        )

        for fold, train_seq, valid_data in sequence.generate():
            model = self.model.create_model()

            if not model._is_compiled:
                print("Model should be compiled.")
                return

            K.set_value(model.optimizer.lr, self.params.learning_rate)

            if callbacks is None:
                callbacks_f = []
            else:
                callbacks_f = callbacks

            cp_callback = ModelCheckpoint(
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
            lg_callback = CSVLogger(
                os.path.join(
                    self.result.histories_dir, "history_fold{}.csv".format(fold)
                )
            )
            callbacks_f.append(cp_callback)
            callbacks_f.append(lg_callback)

            model.fit(
                x=train_seq,
                epochs=self.params.epochs,
                callbacks=callbacks_f,
                validation_data=valid_data,
            )

    def test(self, x: np.ndarray, y: np.ndarray):
        """
        学習したモデルから各foldごとにテストします。

        Args:
            x (np.ndarray): テストするデータ
            y (np.ndarray): データに対応するラベル
        """
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
    ) -> np.ndarray:
        """
        モデルの推定結果を返します。

        Args:
            x (np.ndarray): 推定するデータ
            model_weight_path (str): 推定するときのモデルの重みへのパス

        Returns:
            np.ndarray: 推定結果
        """
        model = self.model.create_model()
        model.load_weights(model_weight_path)

        prediction = model.predict(x)

        return prediction

    def box_plot_history(self, metrics: List[str] = None):
        """
        テスト結果を箱ひげ図にてプロットします

        Args:
            metrics (List[str], optional): プロットする評価値
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
        """
        学習と検証の結果のfoldにおける平均値をプロットします。
        """
        histories = learning_history.from_dir(self.result.histories_dir)
        average = learning_history.average(histories)

        plot.plot_history(average, self.result.average_result_dir)
