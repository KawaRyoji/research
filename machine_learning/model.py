from abc import ABCMeta, abstractmethod
import keras

class learning_model(metaclass=ABCMeta):
    """
    学習させるモデルを定義するクラスです。
    """
    @abstractmethod
    def create_model(self, **kwargs) -> keras.Model:
        """
        モデルの定義とそのモデルを返す関数です。

        Raises:
            NotImplementedError: 実装されてないときのエラー

        Returns:
            keras.Model: 定義したモデル
        """
        raise NotImplementedError()