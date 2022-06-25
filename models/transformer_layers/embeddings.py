import tensorflow as tf
import math

from tensorflow.keras.layers import Dense, Lambda, Layer

class TokenEmbedding(Layer):
    def __init__(self, hidden_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dense = Dense(hidden_dim, use_bias=False)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        return self.dense(input)


class AddPositionalEncoding(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer = Lambda(self._add_positional_encoding)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        return self.layer(input)

    def _add_positional_encoding(self, input: tf.Tensor) -> tf.Tensor:
        i_type = input.dtype
        batch_size, length, dim = tf.unstack(tf.shape(input))

        # NOTE: expend_dims(x, 0)とすると[x]から[1, shape(x)]となる
        #       これに加えてtile(expand_dims(x, 0), [nums, 1])とすると[nums, shape(x)]となり
        #       テンソルxを縦にnums回並べたテンソルとなる

        dim_counter = tf.range(dim) // 2 * 2  # 0, 0, 2, 2, 4, ...
        dim_matrix = tf.tile(tf.expand_dims(dim_counter, 0), [length, 1])
        dim_matrix = tf.pow(10000.0, tf.cast(dim_matrix / dim, i_type))

        phase = (
            tf.cast(tf.range(dim) % 2, i_type) % math.pi / 2
        )  # 0, pi/2, 0, pi/2, ...
        phase_matrix = tf.tile(tf.expand_dims(phase, 0), [length, 1])

        pos_counter = tf.range(length)
        pos_matrix = tf.cast(tf.tile(tf.expand_dims(pos_counter, 1), [1, dim]), i_type)

        positional_encoding = tf.sin(pos_matrix / dim_matrix + phase_matrix)
        positional_encoding = tf.tile(
            tf.expand_dims(positional_encoding, 0), [batch_size, 1, 1]
        )

        return input + positional_encoding
