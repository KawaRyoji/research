import os
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from tensorflow.keras.utils import Sequence
from util.calc import calc_split_point
from util.path import dir2paths


class dataset:
    """
    データセットを扱うクラスです。
    データとラベルのディレクトリまたはパスのリストとデータ生成の関数を与えて使用します。
    """

    def __init__(
        self,
        data_paths: List[str],
        label_paths: List[str],
        construct_process: Callable[[str, str], Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """
        Args:
            data_paths (List[str]): データへのパスのリスト
            label_paths (List[str]): データに対応するラベルへのパスのリスト
            construct_process (Callable[[str, str], Tuple[np.ndarray, np.ndarray]]): パスからデータとラベルを読み込み、モデルに入力する特徴量を返す関数

        Raises:
            RuntimeError: データパスのリストとラベルパスのリストの長さが同じではないときに発生します。
        """
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
        """
        Args:
            data_dir (str): データのディレクトリへのパス
            label_dir (str): データに対応するラベルのディレクトリへのパス
            construct_process (Callable[[str, str], Tuple[np.ndarray, np.ndarray]]): パスからデータとラベルを読み込み、モデルに入力する特徴量を返す関数

        Returns:
            dataset: データセットのインスタンス
        """
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
        """
        データセットをパスとconstruct_processに従って構築し、npzファイルに保存します。

        Args:
            file_name (str): 保存するデータセットのファイル名(拡張子なし)
            limit (int, optional): 読み込むデータの数を制限します
            seed (int, optional): 制限する際のシャッフルに用いられます
            normalize (bool, optional): 標準化をするかどうかを指定します
        """
        if os.path.exists(file_name + ".npz"):
            print(file_name + ".npz", "is already exists")
            return

        data_paths, label_paths = dataset.limit_data(
            self.data_paths, self.label_paths, limit=limit, seed=seed
        )

        print("start construction")
        Path.mkdir(Path(file_name).parent, parents=True, exist_ok=True)

        data_list = []
        label_list = []
        for data_path, labels_path in zip(data_paths, label_paths):
            print("processing:\n\t", data_path, "\n\t", labels_path)
            data, label = self._construct_process(data_path, labels_path, **kwargs)
            data_list.extend(data)
            label_list.extend(label)
            print("processing is complete")

        data_list = np.array(data_list, dtype=np.float32)
        label_list = np.array(label_list, dtype=np.float32)

        # 標準化処理
        if normalize:
            data_list = dataset.normalize_data(data_list)

        np.savez(file_name, x=data_list, y=label_list)
        print(
            "construction is complete\n\tdata shape:\t",
            data_list.shape,
            "\n\tlabels shape:\t",
            label_list.shape,
        )

    def load(self, file_name: str, shuffle=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        データセットを読み込みます。

        Args:
            file_name (str): 読み込むデータセットへのパス(拡張子なし)
            shuffle (bool, optional): データをシャッフルするかどうかを指定します

        Returns:
            np.ndarray, np.ndarray: 読み込んだデータとラベル
        """
        data = np.load(file_name + ".npz", allow_pickle=True)
        x, y = data["x"], data["y"]
        del data

        if shuffle:
            dataset.shuffle_data(x, y)

        return x, y

    @classmethod
    def limit_data(
        cls,
        data_paths: np.ndarray,
        label_paths: np.ndarray,
        limit: int = None,
        seed=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if limit is not None:
            if limit <= 0:
                return data_paths, label_paths

            limit = min(limit, len(data_paths))

            if seed is None:
                seed = np.random.randint(np.iinfo(np.int32).max)

            np.random.seed(seed)
            index = np.random.permutation(len(data_paths))
            data_paths = data_paths[index]
            label_paths = label_paths[index]
            data_paths = data_paths[:limit]
            label_paths = label_paths[:limit]
        else:
            data_paths = data_paths
            label_paths = label_paths

        return data_paths, label_paths

    @classmethod
    def split_data(
        cls, x: np.ndarray, y: np.ndarray, validation_split: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if validation_split <= 0 or validation_split >= 1:
            raise RuntimeError("assuming 0 < split < 1")

        split_point = calc_split_point(len(x), 1 - validation_split)

        return (
            x[:split_point],
            y[:split_point],
            x[split_point:],
            y[split_point:],
        )

    @classmethod
    def shuffle_data(cls, x, y, seed: int = None):
        """
        データとラベルを対応付けてシャッフルします

        Args:
            x (ArrayLike): シャッフルするデータ
            y (ArrayLike): シャッフルするラベル
            seed (int, optional): シャッフルする際のシード値
        """
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)

        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)

    @classmethod
    def normalize_data(cls, data):
        data -= np.mean(data, axis=1)[:, np.newaxis]
        std = np.std(data, axis=1)[:, np.newaxis]
        return np.divide(data, std, out=np.zeros_like(data), where=std != 0)


class data_sequence(Sequence):
    """
    model.fit()または、model.fit_generator()で学習させるデータセットのクラス

    エポックごとにデータのシャッフル処理が入ります

    1エポックのサンプル数は`バッチサイズ x エポックサイズ`となります

    Args:
        data (np.ndarray): データの配列
        labels (np.ndarray): ラベルの配列
        batch_size (int): バッチサイズ
        batches_per_epoch (int): エポックサイズ
            指定されていない場合 `len(data) // batch_size`で計算されます
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
