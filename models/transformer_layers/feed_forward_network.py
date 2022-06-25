import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Layer


class FeedForwardNetwork(Layer):
    def __init__(
        self, hidden_dim: int, ffn_dim: int, dropout_rate: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        self.input_dense = Dense(ffn_dim, activation="relu")
        self.dropout = Dropout(dropout_rate)
        self.output_dense = Dense(hidden_dim)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        output = self.input_dense(input)
        output = self.dropout(output)
        return self.output_dense(output)
