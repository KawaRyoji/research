from abc import ABCMeta, abstractmethod
import keras

class learning_model(metaclass=ABCMeta):
    @abstractmethod
    def create_model(self, **kwargs) -> keras.Model:
        raise NotImplementedError()