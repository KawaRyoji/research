import os
from pathlib import Path
from keras.utils.data_utils import Sequence
import numpy as np
from typing import List, Tuple, Callable
from util.calc import calc_split_point
from util.path import dir2paths


class dataset:
    def __init__(
        self,
        data_paths: List[str],
        label_paths: List[str],
        construct_process: Callable[[str, str], Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        if len(data_paths) != len(label_paths):
            raise RuntimeError(
                "the length of data_paths must match the length of label_paths"
            )

        self.data_paths = np.array(data_paths)
        self.label_paths = np.array(label_paths)
        self._construct_process = construct_process

    @classmethod
    def from_dir(
        cls,
        data_dir: str,
        label_dir: str,
        construct_process: Callable[[str, str], Tuple[np.ndarray, np.ndarray]],
    ):
        data_paths = sorted(dir2paths(data_dir))
        label_paths = sorted(dir2paths(label_dir))

        return cls(data_paths, label_paths, construct_process)

    def construct(
        self,
        file_name: str,
        limit: int = None,
        seed: int = None,
        normalize=False,
        **kwargs
    ):
        if os.path.exists(file_name + ".npz"):
            print(file_name + ".npz", "is already exists")
            return
 
        if limit is not None:
            assert limit >= 1

            limit = min(limit, len(self.data_paths))

            if seed is None:
                seed = np.random.randint(np.iinfo(np.int32).max)

            np.random.seed(seed)
            index = np.random.permutation(len(self.data_paths))
            data_paths = self.data_paths[index]
            label_paths = self.label_paths[index]
            data_paths = data_paths[:limit]
            label_paths = label_paths[:limit]
        else:
            data_paths = self.data_paths
            label_paths = self.label_paths
        
        print("start construction")
        Path.mkdir(Path(file_name).parent, parents=True, exist_ok=True)

        datas = []
        labels = []
        for data_path, labels_path in zip(data_paths, label_paths):
            print("processing:\n\t", data_path, "\n\t", labels_path)
            data, label = self._construct_process(data_path, labels_path, **kwargs)
            datas.extend(data)
            labels.extend(label)
            print("processing is complete")

        datas = np.array(datas, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        # 標準化処理
        if normalize:
            datas = dataset.normalize_data(datas)
            # 0割のデータを除外する
            idx = ~np.all(datas == 0, axis=1)
            datas = datas[idx]
            labels = labels[idx]

        np.savez(file_name, x=datas, y=labels)
        print(
            "construction is complete\n\tdata shape:\t",
            datas.shape,
            "\n\tlabels shape:\t",
            labels.shape,
        )

    def load(self, file_name: str, shuffle=True, validation_split=None):
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

        data = np.load(file_name + ".npz", allow_pickle=True)
        x, y = data["x"], data["y"]
        del data

        if shuffle:
            dataset.shuffle_data(x, y)

        if validation_split is None or validation_split <= 0 or validation_split >= 1:
            return x, y

        split_point = calc_split_point(len(x), 1 - validation_split)

        return (
            x[:split_point],
            y[:split_point],
            x[split_point:],
            y[split_point:],
        )

    @classmethod
    def shuffle_data(cls, x, y, seed=None):
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

    @classmethod
    def normalize_data(cls, data):
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
        dataset.shuffle_data(self.data, self.labels)
