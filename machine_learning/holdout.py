import os
from pathlib import Path
import keras
import keras.backend as K
import numpy as np
from machine_learning.dataset import data_sequence, load, normalize_data
from machine_learning.interfaces import learning_model
from machine_learning.learning_history import learning_history
from machine_learning.parameter import hyper_params
import machine_learning.plot as plot

# TODO: 新しいプログラムへの対応
class holdout:
    """
    hold-out法によりモデルを学習させるインタフェースです

    Imachine_learningに依存しています
    """

    def __init__(
        self,
        ml: learning_model,
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

        if not model._is_compiled:
            print("Model should be compiled.")
            return
        K.set_value(model.optimizer.lr, self.params.learning_rate)

        print(model.summary())
        print("loading dataset ...")
        train_x, train_y, valid_x, valid_y = load(
            train_set_path, validation_split=self.validation_split
        )
        print("loading dataset is done")
        print("train data shape:", train_x.shape)
        print("train labels shape:", train_y.shape)
        print("validation data shape:", valid_x.shape)
        print("validation labels shape:", valid_y.shape)
        print(self.params)

        result_dir = os.path.join(self.ml.output_dir, "holdout")
        history_path = os.path.join(result_dir, "history.csv")
        model_weights_dir = os.path.join(result_dir, "model_weights")
        checkpoint_path = os.path.join(model_weights_dir, "cp_best.ckpt")
        figures_dir = os.path.join(result_dir, "figures")

        Path.mkdir(Path(result_dir), parents=True, exist_ok=True)
        Path.mkdir(Path(model_weights_dir), exist_ok=True)

        self.params.save_to_json(os.path.join(result_dir, "params.json"))

        if callbacks is None:
            callbacks = []

        cpb_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor_best_cp,
            mode=monitor_mode,
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
        )
        lg_callback = keras.callbacks.CSVLogger(history_path)
        callbacks.append(cpb_callback)
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

        history = learning_history.from_path(history_path)

        if save_figure:
            plot.plot_history(
                history,
                figures_dir,
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
        model.load_weights(model_weight_path)

        res = ""
        for metric, val in zip(model.metrics_names, model.evaluate(x, y, verbose=2)):
            res += "{}: {}\n".format(metric, val)

        Path.mkdir(
            Path(os.path.join(self.ml.output_dir, "hold_out", "test_res.txt")).parent,
            parents=True,
            exist_ok=True,
        )
        with open(
            os.path.join(self.ml.output_dir, "hold_out", "test_res.txt"), mode="w"
        ) as f:
            f.write(res)

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

            model.load_weights(model_weight_path)

        res = ""
        for metric, val in zip(model.metrics_names, model.evaluate(x, y, verbose=2)):
            res += "{}: {}\n".format(metric, val)

        Path.mkdir(Path(save_txt_path).parent, parents=True, exist_ok=True)
        with open(save_txt_path, mode="w") as f:
            f.write(res)

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
