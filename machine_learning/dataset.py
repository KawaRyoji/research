import os
from pathlib import Path
from keras.utils.data_utils import Sequence
import numpy as np

from util.calc import calc_split_point


def construct_dataset(
    data_paths: list,
    labels_paths: list,
    output_path: str,
    process,
    normalize=True,
    **kwargs
):
    """
    機械学習のデータセットを構築し、.npzファイルに保存します

    ## Params
        - data_paths (list): データのパスのリスト
        - labels_paths (list): ラベルのパスのリスト\n

        データとラベルのパスが対応するようにlistに格納してください

        - output_path (str): 保存先のパス
            - ファイル名は拡張子(.npz)を付けずに指定してください
        - process (function(data_path, labels_path, **kwargs)): データとラベル1組に対する処理
            - params
                - data_path (str): データのパス
                - labels_path (str): ラベルのパス
                - **kwargs: 処理に渡す引数
            - returns
                - datas (list): データ(音データ、画像データなど)
                - labels (list): データに対応するラベル
        - normalize (bool, optional): Trueの場合、データを標準化します\n
            推論する場合は同じくデータを標準化してください
    """

    if os.path.exists(output_path + ".npz"):
        print(output_path + ".npz", "is already exists")
        return

    if len(data_paths) != len(labels_paths):
        raise "the length of data_paths must match the length of labels_paths"

    print("start construction")
    Path.mkdir(Path(output_path).parent, parents=True, exist_ok=True)

    datas = []
    labels = []
    for data_path, labels_path in zip(data_paths, labels_paths):
        print("processing:\n\t", data_path, "\n\t", labels_path)
        data, label = process(data_path, labels_path, **kwargs)
        datas.extend(data)
        labels.extend(label)
        print("processing is complete")

    datas = np.array(datas, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # 標準化処理
    if normalize:
        datas = normalize_data(datas)
        # 0割のデータを除外する
        idx = ~ np.all(datas==0, axis=1)
        datas = datas[idx]
        labels = labels[idx]

    np.savez(output_path, x=datas, y=labels)
    print(
        "construction is complete\n\tdata shape:\t",
        datas.shape,
        "\n\tlabels shape:\t",
        labels.shape,
    )


def load(data_path: str, shuffle=True, validation_split=None):
    """
    .npzを読み込み、データとラベルのセットを返します

    ## Params
        - data_path (str): 読み込む.npzファイル
        - shuffle (bool): Trueの場合、データをシャッフルして読み込みます
        - validation_split (double): 検証用セットの割合を指定します
            - 0 < validation_split < 1で指定してください

    ## Returns
        - x, y: validation_splitが指定されてない場合
            - x (np.ndarray): データのテンソル
            - y (np.ndarray): ラベルのテンソル
        - train_x, train_y, valid_x, valid_y: validation_splitが指定されている場合
            - train_x (np.ndarray): 学習データのテンソル
            - train_y (np.ndarray): 学習ラベルのテンソル
            - valid_x (np.ndarray): 検証データのテンソル
            - valid_y (np.ndarray): 検証ラベルのテンソル
    """

    data = np.load(data_path, allow_pickle=True)
    x, y = data["x"], data["y"]
    del data

    if shuffle:
        shuffle_data(x, y)

    if validation_split is None or validation_split <= 0 or validation_split >= 1:
        return x, y

    split_point = calc_split_point(len(x), 1 - validation_split)

    return (
        x[:split_point],
        y[:split_point],
        x[split_point:],
        y[split_point:],
    )


def shuffle_data(x, y, seed=None):
    """
    データとラベルの組をシャッフルします

    このメソッドは破壊的処理です

    ## Params
        - x (array): データの配列
        - y (array): ラベルの配列

        x, yは最初の次元のサイズを一致させてください
    """

    if seed is None:
        seed = np.random.randint(np.iinfo(np.int32).max)

    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)


class data_sequence(Sequence):
    """
    model.fit()または、model.fit_generator()で学習させるデータセットのクラス

    エポックごとにデータのシャッフル処理が入ります

    1エポックのサンプル数は`バッチサイズ x エポックサイズ`となります

    ## Constructer Params
        - data (array): データの配列
        - labels (array): ラベルの配列
        - batch_size (int): バッチサイズ
        - batches_per_epoch (int): エポックサイズ
            - 指定されていない場合 `len(data) // batch_size`で計算されます
    """

    def __init__(self, data, labels, batch_size, batches_per_epoch=None) -> None:
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        if batches_per_epoch is None:
            self.batches_per_epoch = len(self.data) // self.batch_size
        else:
            self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        return self.data[start:end], self.labels[start:end]

    def on_epoch_end(self):
        shuffle_data(self.data, self.labels)


def normalize_data(data):
    """
    データを標準化します\n
    0割が発生する場合、そのデータは全0になります
        
    ## Params
        - datas (array): 標準化するデータ

    ## Returns
        - array: 標準化されたデータ
    """
    data -= np.mean(data, axis=1)[:, np.newaxis]
    std = np.std(data, axis=1)[:, np.newaxis]
    return np.divide(data, std, out=np.zeros_like(data), where=std != 0)
    
