from abc import ABCMeta, abstractmethod
import os
import keras
import numpy as np

from machine_learning.dataset import construct_dataset, load, normalize_data
from machine_learning.model import hyper_params
import machine_learning.plot as plot
from util.path import dir2paths
from machine_learning import dataset
from machine_learning import model as m


class Imachine_learning(metaclass=ABCMeta):
    """
    機械学習を行うためのモデルの定義とデータセットの構築をするためのインタフェース
    """

    def __init__(
        self,
        train_data_paths: list,
        train_labels_paths: list,
        test_data_paths: list,
        test_labels_paths: list,
        output_dir: str,
    ) -> None:
        """
        ## Params
            - train_data_paths (list): 学習データのパスのリスト
            - train_labels_paths (list): 学習ラベルのパスのリスト
            - test_data_paths (list): テストデータのパスのリスト
            - test_labels_paths (list): テストラベルのパスのリスト
            - output_dir (str): すべての出力の保存先ディレクトリパス
        """
        self.output_dir = output_dir
        self.train_data_paths = np.array(train_data_paths)
        self.train_labels_paths = np.array(train_labels_paths)
        self.test_data_paths = np.array(test_data_paths)
        self.test_labels_paths = np.array(test_labels_paths)

    @classmethod
    def from_dir(
        cls,
        train_data_dir: str,
        train_labels_dir: str,
        test_data_dir: str,
        test_labels_dir: str,
        output_dir: str,
    ):
        """
        ディレクトリパスからインスタンスを生成します

        ## Params
            - train_data_dir (str):学習データのディレクトリパス
            - train_labels_dir (str):学習ラベルのディレクトリパス
            - test_data_dir (str):テストデータのディレクトリパス
            - test_labels_dir (str):テストラベルのディレクトリパス
            - output_dir (str): すべての出力の保存先ディレクトリパス

        ## Returns
            - Imachine_learning: インスタンス
        """
        train_data_paths = dir2paths(train_data_dir)
        train_labels_paths = dir2paths(train_labels_dir)
        test_data_paths = dir2paths(test_data_dir)
        test_labels_paths = dir2paths(test_labels_dir)

        return cls(
            train_data_paths,
            train_labels_paths,
            test_data_paths,
            test_labels_paths,
            output_dir,
        )

    @abstractmethod
    def create_model(self) -> keras.Model:
        """
        モデルを記述する関数\n
        モデルはコンパイルを行ってください

        ## Returns
            - keras.Model: 作成したモデル
        """
        raise NotImplementedError()

    @abstractmethod
    def _create_dataset_process(self, data_path: str, label_path: str) -> tuple:
        """
        データとラベル1組に対する処理

        ## Params
            - data_path (str): データのパス
            - labels_path (str): ラベルのパス

        ## Returns
            - datas (list): データ(音データ、画像データなど)
            - labels (list): データに対応するラベル
        """
        raise NotImplementedError()

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

        if limit is not None:
            assert limit >= 1
            limit = min(limit, len(self.train_data_paths))
            if seed is None:
                seed = np.random.randint(np.iinfo(np.int32).max)
            np.random.seed(seed)
            index = np.random.permutation(len(self.train_data_paths))
            self.train_data_paths = self.train_data_paths[index]
            self.train_labels_paths = self.train_labels_paths[index]
            self.train_data_paths = self.train_data_paths[:limit]
            self.train_labels_paths = self.train_labels_paths[:limit]

        if output_path is None:
            output_path = os.path.join(self.output_dir, "train")

        construct_dataset(
            self.train_data_paths,
            self.train_labels_paths,
            output_path,
            self._create_dataset_process,
            **kwargs,
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

        if limit is not None:
            assert limit >= 1
            limit = min(limit, len(self.test_data_paths))
            if seed is None:
                seed = np.random.randint(np.iinfo(np.int32).max)
            np.random.seed(seed)
            index = np.random.permutation(len(self.test_data_paths))
            self.test_data_paths = self.test_data_paths[index]
            self.test_labels_paths = self.test_labels_paths[index]

        if output_path is None:
            output_path = os.path.join(self.output_dir, "test")

        construct_dataset(
            self.test_data_paths,
            self.test_labels_paths,
            output_path,
            self._create_dataset_process,
            **kwargs,
        )