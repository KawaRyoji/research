import tensorflow as tf

from tensorflow.keras.layers import Dropout, Add, Layer, LayerNormalization


class ResidualNormalizationWrapper(Layer):
    def __init__(self, layer: Layer, dropout_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.layer_normalization = LayerNormalization()
        self.dropout = Dropout(dropout_rate)
        self.add = Add()

    def call(self, input: tf.Tensor, **kwargs) -> tf.Tensor:
        output = self.layer_normalization(input)
        output = self.layer(output, **kwargs)
        output = self.dropout(output)

        return self.add([input, output])
