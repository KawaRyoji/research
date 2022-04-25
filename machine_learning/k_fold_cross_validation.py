import os

import numpy as np
from machine_learning.dataset import load, normalize_data
from machine_learning.interfaces import Imachine_learning
from machine_learning.model import hyper_params
import machine_learning.model as m
import machine_learning.plot as plot


class k_fold_cross_validation:
    """
    k分割交差検証により学習させるインタフェースです

    Imachine_learningに依存しています
    """

    def __init__(
        self,
        ml: Imachine_learning,
        params: hyper_params,
        k,
    ) -> None:
        self.ml = ml
        self.params = params
        self.k = k
        self.result_dir = os.path.join(ml.output_dir, "result_{}fold".format(k))

    def train(
        self,
        train_set_path: str = None,
        save_figure=True,
        monitor_best_cp="val_loss",
        monitor_mode="auto",
        callbacks=None,
        valid_size=None,
    ):
        """
        k分割交差検証での学習と検証を行います

        ## Params:
            - train_set_path (str, optional): 指定されていない場合output_dir/train.npzが学習に使用されます
            - save_figure (bool, optional): Trueの場合結果をグラフにプロットします
            - monitor_best_cp (str, optional): 監視するログの変数名
            - monitor_mode (str, optional): 監視のモード
            - callbacks (list, optional): 追加するコールバック関数
                - 必ず各foldごとのmetricsのログと各foldでもっとも性能が高い重みが保存されます
            - valid_size (int, optional): 検証セットのサイズを指定できます
        """
        if train_set_path is None:
            train_set_path = os.path.join(self.ml.output_dir, "train.npz")

        m.k_fold_cross_validation_train(
            self.ml.create_model,
            params=self.params,
            train_data_path=train_set_path,
            k=self.k,
            save_result_dir=self.result_dir,
            save_figure=save_figure,
            monitor_best_cp=monitor_best_cp,
            monitor_mode=monitor_mode,
            callbacks=callbacks,
            valid_size=valid_size,
        )

    def test(self, data_path: str = None):
        """
        k分割交差検証でテストを行います

        ## Params:
            - data_path (str, optional): 指定されていない場合output_dir/test.npzが使用されます
        """

        if data_path is None:
            data_path = os.path.join(self.ml.output_dir, "test.npz")

        x, y = load(data_path, shuffle=False)
        m.k_fold_cross_validation_test(
            self.ml.create_model,
            self.k,
            os.path.join(self.result_dir, "model_weights/cp_best_fold{}.ckpt"),
            os.path.join(self.result_dir, "test_res.csv"),
            x,
            y,
        )

    def predict(
        self,
        model_weight_path: str,
        data_path: str = None,
        save_fig=True,
    ):
        """
        モデルの重みをロードし、テストデータで推論します

        ## Params
            - model_weight_path (str): 指定している場合、指定されたモデルの重みを用います
            - data_path (str, optional): テストするデータのパス(.npzファイル)\n
                Noneが指定されている場合、output_dir/test.npzが読み込まれます\n
            - save_fig (bool, optional): Trueの場合、活性化マップとラベルを重ねたグラフを保存します

        ## Returns
            prediction: 推定マップ
        """

        model = self.ml.create_model()
        model.load_weights(model_weight_path)

        if data_path is None:
            data_path = os.path.join(self.ml.output_dir, "test.npz")

        x, y = load(data_path, shuffle=False)
        prediction = model.predict(x)

        if save_fig:
            plot.plot_activation_with_labels(
                y,
                prediction,
                os.path.join(
                    self.result_dir,
                    "figures",
                    "activation_with_labels_testset.png",
                ),
            )

        return prediction

    def predict_from_raw_data(
        self,
        data_path: str,
        label_path: str,
        model_weight_path: str,
        save_fig=True,
        save_fig_path=None,
        threshold=None,
        normalize=True,
    ):
        """
        前処理を行っていないデータでの推論を行います


        ## Params
            - data_path (str): 前処理を行っていないデータのパス
            - label_path (str): 前処理を行っていないラベルのパス
            - model_weight_path (str): 指定されたモデルの重みを用います
            - save_fig (bool, optional): Trueの場合、活性化マップとラベルを重ねたグラフを保存します
            - save_fig_path (str, optional): 指定している場合、画像の保存先を指定したファイルにします\n
                指定していない場合、output_dir/figures/activation_with_labels_{filename}.png に保存されます
            - threshold (float, optional): 指定している場合、活性化マップをしきい値以上のマップとして保存します
        """
        model = self.ml.create_model()
        model.load_weights(model_weight_path)

        x, y = self.ml._create_dataset_process(data_path, label_path)
        if normalize:
            x = normalize_data(x)
        prediction = model.predict(x)

        if threshold is not None:
            prediction = np.where(prediction < threshold, 0, 1)

        if save_fig:
            if save_fig_path is None:
                save_fig_path = os.path.join(
                    self.result_dir,
                    "figures",
                    "activation_with_labels_"
                    + os.path.splitext(os.path.basename(data_path))[0]
                    + ".png",
                )

            plot.plot_activation_with_labels(
                y,
                prediction,
                save_fig_path,
            )

        return prediction

    def create_train_set(
        self, limit: int = None, seed: int = None, output_path: str = None, **kwargs
    ) -> None:
        """
        学習データセットを作成します\n
        limitが指定されている場合、ファイルはシャッフルされてlimitの数だけ読み込みます\n

        ## Params
            - limit (int, optional): 用いるファイルの数を制限します\n
                Noneが指定されている場合、すべてのファイルを指定します

            - seed (int, optional): シャッフルのシード値\n
                Noneが指定されている場合、ランダムに選出されます

            - output_path (str, optional): データセットの保存先のパス\n
                Noneが指定されている場合、output_dir/train.npzとなります\n
                指定する場合は拡張子を付けないでください
        """
        self.ml.create_train_set(
            limit=limit, seed=seed, output_path=output_path, **kwargs
        )

    def create_test_set(
        self, limit: int = None, seed: int = None, output_path: str = None, **kwargs
    ) -> None:
        """
        テストデータセットを作成します\n
        limitが指定されている場合、ファイルはシャッフルされてlimitの数だけ読み込みます\n

        ## Params
            - limit (int, optional): 用いるファイルの数を制限します\n
                Noneが指定されている場合、すべてのファイルを指定します

            - seed (int, optional): シャッフルのシード値\n
                Noneが指定されている場合、ランダムに選出されます

            - output_path (str, optional): データセットの保存先のパス\n
                Noneが指定されている場合、output_dir/test.npzとなります\n
                指定する場合は拡張子を付けないでください
        """
        self.ml.create_test_set(
            limit=limit, seed=seed, output_path=output_path, **kwargs
        )

    def box_plot_history(self, metrics: list = None):
        """
        テスト結果を箱ひげ図にてプロットします

        ## Params:
            metrics (list, optional): プロットする評価値を指定できます
        """
        plot.box_plot_history(
            os.path.join(
                self.result_dir,
                "figures",
                "test_res.png",
            ),
            os.path.join(
                self.result_dir, "test_res.csv"
            ),
            metrics=metrics,
        )

    def plot_average_history(self):
        plot.plot_average_history(
            os.path.join(
                self.result_dir,
                "figures",
                "kcv_result_average",
            ),
            os.path.join(
                self.result_dir, "histories"
            ),
        )

    @classmethod
    def plot_compare_average_history(
        cls,
        output_dir: str,
        k: int,
        *models: Imachine_learning,
        legend: list = None,
        metrics: list = None,
    ):
        """
        検証結果の平均を比較してプロットします

        ## Params:
            - output_dir (str): 結果を保存するディレクトリパス
            - k (int): 使用したk
            - models (Imachine_learning): 比較するモデル
            - legend (list, optional): 凡例を指定できます
            - metrics (list, optional): 比較する評価値を指定できます
        """
        plot.plot_compare_average_history(
            output_dir,
            *list(
                map(
                    lambda x: os.path.join(
                        x.output_dir, "result_{}fold".format(k), "histories"
                    ),
                    models,
                ),
            ),
            legend=legend,
            metrics=metrics,
        )

    @classmethod
    def box_plot_history_compare(
        cls,
        output_path: str,
        k: int,
        *models: Imachine_learning,
        stripplot=False,
        legend: list = None,
        metrics: list = None,
    ):
        """
        箱ひげ図によりテスト結果を比較しプロットします

        ## Params:
            - output_dir (str): 結果を保存するディレクトリパス
            - k (int): 使用したk
            - stripplot (bool, optional): Trueの場合、散布図を重ねてプロットします
            - legend (list, optional): 凡例を指定できます
            - metrics (list, optional): 比較する評価値を指定できます
        """
        plot.box_plot_history_compare(
            output_path,
            *list(
                map(
                    lambda x: os.path.join(
                        x.output_dir, "result_{}fold".format(k), "test_res.csv"
                    ),
                    models,
                ),
            ),
            stripplot=stripplot,
            legend=legend,
            metrics=metrics,
        )
