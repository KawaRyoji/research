import os

import numpy as np
from machine_learning.dataset import load, normalize_data
from machine_learning.interfaces import Imachine_learning
from machine_learning.model import hyper_params
import machine_learning.plot as plot
import machine_learning.model as m

class Hold_out:
    """
    hold-out法によりモデルを学習させるインタフェースです

    Imachine_learningに依存しています
    """

    def __init__(
        self,
        ml: Imachine_learning,
        params: hyper_params,
        validation_split: float,
    ) -> None:
        assert validation_split > 0 and validation_split < 1
        self.ml = ml
        self.params = params
        self.validation_split = validation_split

    def train(
        self,
        save_figure=True,
        train_set_path: str = None,
        monitor_best_cp="val_loss",
        monitor_mode="auto",
        callbacks=None,
    ) -> None:
        """
        モデルを学習させます

        ## Params
            - save_figure (bool, optional): Trueの場合学習後のmetricsのグラフを保存します\n
                デフォルト値はTrueです
            - train_set_path (str, optional): 用いる学習データセットのパス\n
                Noneが指定されている場合、output_dir/train.npzとなります
            - callbacks (array, optional): 指定されている場合、このコールバック関数が学習中に呼び出されます.
            指定されていない場合、モデルの重み、metricsのログがコールバックとして指定されます
        """

        model = self.ml.create_model()

        if train_set_path is None:
            train_set_path = os.path.join(self.ml.output_dir, "train.npz")

        m.train_model(
            model,
            self.params,
            self.validation_split,
            train_set_path,
            self.ml.output_dir + "/hold_out",
            save_figure=save_figure,
            monitor_best_cp=monitor_best_cp,
            monitor_mode=monitor_mode,
            callbacks=callbacks,
        )

    def test(
        self,
        data_path: str = None,
        model_weight_path: str = None,
    ):
        """
        モデルをテストし、その評価値を保存します

        ## Params
            - data_path (str, optional): テストするデータのパス(.npzファイル)\n
                Noneが指定されている場合、output_dir/test.npzが読み込まれます\n
            - epoch (int, optional): 指定している場合、指定されたエポックの重みを用います
            - model_weight_path (str, optional): 指定している場合、指定されたモデルの重みを用います\n
                指定されていない場合、cp_best.ckptを使用します
        """

        model = self.ml.create_model()
        if data_path is None:
            data_path = os.path.join(self.ml.output_dir, "test.npz")

        if model_weight_path is None:
            model_weight_path = os.path.join(
                self.ml.output_dir, "hold_out", "model_weights", "cp_best.ckpt"
            )

        x, y = load(data_path, shuffle=False)
        m.test_model(
            model,
            model_weight_path,
            os.path.join(self.ml.output_dir, "hold_out", "test_res.txt"),
            x,
            y,
        )

    def test_from_raw_data(
        self,
        data_path: str,
        labels_path: str,
        model_weight_path: str = None,
        save_txt_path: str = None,
        normalize=True,
    ):
        """
        前処理を行っていないデータでモデルをテストし、その評価値を表示します

        ## Params
            - data_path (str): 前処理を行っていないデータのパス
            - label_path (str): 前処理を行っていないラベルのパス
            - epoch (int, optional): 指定している場合、指定されたエポックの重みを用います
            - model_weight_path (str, optional): 指定している場合、指定されたモデルの重みを用います\n
                指定されていない場合、cp_best.ckptを使用します
            - save_txt_path (str, optional): 指定している場合、指定されたファイルパスに結果を保存します\n
                Noneが指定されている場合、output_dir/test_res.txtに保存します
            - normalize (bool, optional): Trueの場合データを標準化します
        """

        model = self.ml.create_model()
        x, y = self.ml._create_dataset_process(data_path, labels_path)

        if model_weight_path is None:
            model_weight_path = os.path.join(
                self.ml.output_dir, "hold_out", "model_weights", "cp_best.ckpt"
            )

        if normalize:
            x = normalize_data(x)

        if save_txt_path is None:
            save_txt_path = os.path.join(self.ml.output_dir, "hold_out", "test_res.txt")

        m.test_model(model, model_weight_path, save_txt_path, x, y)

    def predict(
        self,
        model_weight_path: str,
        data_path: str = None,
        save_fig=True,
    ):
        """
        モデルの重みをロードし、テストデータで推論します

        ## Params
            - model_weight_path (str): 指定されたモデルの重みを用います
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
                    self.ml.output_dir,
                    "hold_out",
                    "figures/activation_with_labels_testset.png",
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
            - save_fig_name (str, optional): 指定している場合、画像の保存先を指定したファイルにします\n
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
                    self.ml.output_dir,
                    "hold_out",
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

    @classmethod
    def plot_compare_history(
        cls, output_dir:str, *models: Imachine_learning, legend: list = None
    ):
        """
        各評価値のエポック数による変化を比較しプロットします

        ## Params:
            - output_dir (str): 結果の保存先ディレクトリパス
            - models (Imachine_learning): 比較するモデル
            - legend (list, optional): 凡例を指定できます
        """
        plot.plot_compare_history(
            output_dir,
            *[
                map(
                    lambda x: os.path.join(x.output_dir, "hold_out", "history.csv"),
                    models,
                )
            ],
            legend=legend,
        )
