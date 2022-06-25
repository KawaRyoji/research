from abc import ABCMeta, abstractmethod
from tensorflow.keras import Model


class learning_model(metaclass=ABCMeta):
    """
    学習させるモデルを定義するクラスです。
    """

    @abstractmethod
    def create_model(self, **kwargs) -> Model:
        """
        モデルの定義とそのモデルを返す関数です。

        Raises:
            NotImplementedError: 実装されてないときのエラー

        Returns:
            Model: 定義したモデル
        """
        raise NotImplementedError()
