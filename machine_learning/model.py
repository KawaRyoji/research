from operator import index
import os
from pathlib import Path
import keras
from machine_learning.dataset import data_sequence, load
from machine_learning.metrics import apply_F1_from_log, F1_from_log
import json
from machine_learning import plot
from keras import backend as K
import numpy as np
import pandas as pd

# TODO: オプティマイザをモデルに適用する
class hyper_params:
    """
    学習のハイパーパラメータを保持するクラス

    ## Fields
        - batch_size (int): バッチサイズ
        - epoch_size (int): エポックサイズ\n
            指定されていない場合`int(len(data) / batch_size)`で計算されます
        - epochs (int): エポック数
        - learning_rate (float): 学習率
        - validation_split (float): 検証データの割合\n
            0 < validation_split < 1で指定してください
        - optimizer (str or keras.optimizers.Optimizer): オプティマイザ\n
            学習率を指定する場合はOptimizerクラスを引数としてください\n
            デフォルトはAdamオプティマイザです
    """

    def __init__(
        self,
        batch_size,
        epochs,
        epoch_size=None,
        learning_rate=None,
        optimizer=keras.optimizers.Adam,
    ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.epochs = epochs
        if learning_rate is not None:
            self.optimizer = optimizer(learning_rate=learning_rate)
        else:
            self.optimizer = optimizer

    def __str__(self) -> str:
        return "batch_size:{}\nepochs:{}\nepoch_size:{}\nlearning_rate:{}\noptimizer:{}".format(
            self.batch_size,
            self.epochs,
            self.epoch_size,
            self.learning_rate,
            self.optimizer.__class__.__name__,
        )

    def save_to_json(self, save_path):
        Path.mkdir(Path(save_path).parent, parents=True, exist_ok=True)

        with open(save_path, mode="w") as f:
            json.dump(
                {
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "epoch_size": self.epoch_size,
                    "learning_rate": self.learning_rate,
                    "optimizer": self.optimizer.__class__.__name__,
                },
                f,
            )


def train_model(
    model: keras.Model,
    params: hyper_params,
    validation_split: float,
    train_data_path: str,
    save_result_dir: str,
    save_figure=True,
    monitor_best_cp="val_loss",
    monitor_mode="auto",
    callbacks=None,
):
    """
    モデルを学習させます

    ## Params
        - model (keras.Model): 学習させるモデル
        - params (hyper_params): 学習のハイパーパラメータ
        - validation_split (float): 検証データの割合
        - train_data_path (str): 学習に使用するデータセットのパス(npzファイル)
        - save_result_dir (str): 結果を保存するディレクトリパス
            - モデルのエポックごとの重みと学習履歴が保存されます
        - save_figure (bool, optional): Trueの場合、各metricsをプロットしたものをoutput_dir/figure/に保存します
        - monitor_best_cp (str, optional): 指定した評価値を監視して、最大の性能を出力したモデルの重みを保存します
            - デフォルトはval_lossです
        - callbacks (array, optional): 指定されている場合、このコールバック関数が学習中に呼び出されます.
            callbacksには必ずmetricsのログともっとも性能のよかったモデルの重みの保存が追加されます
    """
    if not model._is_compiled:
        print("Model should be compiled.")
        return
    K.set_value(model.optimizer.lr, params.learning_rate)

    print(model.summary())
    print("loading dataset ...")
    train_x, train_y, valid_x, valid_y = load(
        train_data_path, validation_split=validation_split
    )
    print("loading dataset is done")
    print("train data shape:", train_x.shape)
    print("train labels shape:", train_y.shape)
    print("validation data shape:", valid_x.shape)
    print("validation labels shape:", valid_y.shape)
    print(params)

    Path.mkdir(Path(save_result_dir), parents=True, exist_ok=True)
    Path.mkdir(Path(os.path.join(save_result_dir, "model_weights")), exist_ok=True)

    params.save_to_json(os.path.join(save_result_dir, "params.json"))

    if callbacks is None:
        callbacks = []
        
    cpb_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(save_result_dir, "model_weights", "cp_best.ckpt"),
        monitor=monitor_best_cp,
        mode=monitor_mode,
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    )
    lg_callback = keras.callbacks.CSVLogger(
        os.path.join(save_result_dir, "history.csv")
    )
    callbacks.append(cpb_callback)
    callbacks.append(lg_callback)

    train_sequence = data_sequence(
        train_x, train_y, params.batch_size, batches_per_epoch=params.epoch_size
    )

    del train_x
    del train_y

    model.fit(
        x=train_sequence,
        epochs=params.epochs,
        callbacks=callbacks,
        validation_data=(valid_x, valid_y),
    )
    
    apply_F1_from_log(os.path.join(save_result_dir, "history.csv"))
    
    if save_figure:
        plot.plot_history(
            os.path.join(save_result_dir, "history.csv"),
            os.path.join(save_result_dir, "figure"),
        )


def test_model(
    model: keras.Model,
    model_weight_path: str,
    save_txt_path: str,
    test_data,
    test_labels,
):
    """
    モデルをテストし、その評価値を保存します

    ## Params
        - model (keras.Model): テストするモデル
        - model_weight_path (str): テストするモデルの重みのファイルパス
        - save_txt_path (str): 結果の保存先
        - test_data (array): テストするデータ
        - test_labels (array): テストするデータのラベル
    """
    model.load_weights(model_weight_path)

    res = ""
    for metric, val in zip(
        model.metrics_names, model.evaluate(test_data, test_labels, verbose=2)
    ):
        res += "{}: {}\n".format(metric, val)

    Path.mkdir(Path(save_txt_path).parent, parents=True, exist_ok=True)
    with open(save_txt_path, mode="w") as f:
        f.write(res)


def k_fold_cross_validation_train(
    create_model,
    params: hyper_params,
    train_data_path: str,
    k: int,
    save_result_dir: str,
    save_figure: bool,
    monitor_best_cp="val_loss",
    monitor_mode="auto",
    callbacks=None,
    valid_size=None,
):
    """
    k分割交差検証で学習を行います

    ## Params:
        - create_model (function): モデルを生成する関数
        - params (hyper_params): 学習のハイパーパラメータ
        - train_data_path (str): 学習で用いるデータセット(npzファイル)
        - k (int): 何分割で行うか
        - save_result_dir (str): 結果の保存先ディレクトリパス
        - save_figure (bool): Trueの場合結果をグラフにプロットします
        - monitor_best_cp (str, optional): 監視するログの変数名
        - monitor_mode (str, optional): 監視のモード
        - callbacks (list, optional): 追加するコールバック関数
            - 必ず各foldごとのmetricsのログと各foldでもっとも性能が高い重みが保存されます
        - valid_size (int, optional): 検証セットのサイズを指定できます
    """

    print("loading dataset ...")
    x, y = load(train_data_path)
    print("loading dataset is done")
    print("train data shape:", x.shape)
    print("train labels shape:", y.shape)
    print(params)

    Path.mkdir(Path(save_result_dir), parents=True, exist_ok=True)
    Path.mkdir(Path(os.path.join(save_result_dir, "model_weights")), exist_ok=True)
    Path.mkdir(Path(os.path.join(save_result_dir, "histories")), exist_ok=True)
    Path.mkdir(Path(os.path.join(save_result_dir, "figures")), exist_ok=True)

    params.save_to_json(os.path.join(save_result_dir, "params.json"))

    fold_size = len(x) // k
    for fold in range(k):
        model: keras.Model = create_model()
        if not model._is_compiled:
            print("Model should be compiled.")
            return
        K.set_value(model.optimizer.lr, params.learning_rate)

        if callbacks is None:
            callbacks_f = []
        else:
            callbacks_f = callbacks

        cp_callback = keras.callbacks.ModelCheckpoint(
            os.path.join(
                save_result_dir,
                "model_weights",
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
                save_result_dir, "histories", "history_fold{}.csv".format(fold)
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
            x_train, y_train, params.batch_size, batches_per_epoch=params.epoch_size
        )

        if valid_size is not None:
            x_valid = x_valid[:valid_size]
            y_valid = y_valid[:valid_size]

        model.fit(
            x=sequence,
            epochs=params.epochs,
            callbacks=callbacks_f,
            validation_data=(x_valid, y_valid),
        )

        apply_F1_from_log(
            os.path.join(
                save_result_dir, "histories", "history_fold{}.csv".format(fold)
            )
        )

        if save_figure:
            Path.mkdir(
                Path(
                    os.path.join(
                        save_result_dir, "figures", "figure_fold{}".format(fold)
                    )
                ),
                exist_ok=True,
            )
            plot.plot_history(
                os.path.join(
                    save_result_dir, "histories", "history_fold{}.csv".format(fold)
                ),
                os.path.join(save_result_dir, "figures", "figure_fold{}".format(fold)),
            )


def k_fold_cross_validation_test(
    create_model,
    k: int,
    model_weights_path: str,
    save_result_path: str,
    test_data,
    test_labels,
):
    """
    k分割交差検証でテストを行います

    ## Params:
        create_model (function): モデルを生成する関数
        k (int): 何分割で行うか
        model_weights_path (str): 使用するモデルの重み
        save_result_path (str): 結果の保存先
        test_data (np.ndarray): テストデータ
        test_labels (np.ndarray): テストラベル
    """
    results = []
    for fold in range(k):
        model: keras.Model = create_model()
        model.load_weights(model_weights_path.format(fold))
        res = model.evaluate(test_data, test_labels, verbose=2)
        results.append(res)

    df = pd.DataFrame(
        results,
        index=["fold{}".format(i) for i in range(k)],
        columns=model.metrics_names,
    )

    df = F1_from_log(df)
    df.to_csv(save_result_path)


def get_activation(model: keras.Model, model_weight_path: str, test_data_path: str):
    """
    モデルをテストし、その活性化マップを返します\n
    活性化マップは転置された状態で返されます

    ## Params
        - model (keras.Model): テストするモデル
        - model_weight_path (str): テストするモデルの重みのファイルパス
        - test_data_path (str): テストするデータのパス

    ## Returns
        - np.ndarray: 活性化マップ
    """
    x, _ = load(test_data_path, shuffle=False)
    model.load_weights(model_weight_path)
    return model.predict(x).T
