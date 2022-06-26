import tensorflow as tf

from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.optimizers import Adam
from machine_learning.metrics import F1
from machine_learning.model import learning_model
from models.transformer_layers.encoder import Encoder


class Transformer(learning_model):
    def __init__(
        self,
        data_length: int = 16,
        encoder_input_dim: int = 1024,
        hidden_dim: int = 512,
        ffn_dim: int = 2048,
        output_dim: int = 128,
        num_layer: int = 6,
        num_head: int = 8,
        dropout_rate: float = 0.1,
    ) -> None:
        self.data_length = data_length
        self.encoder_input_dim = encoder_input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.num_head = num_head
        self.dropout_rate = dropout_rate

    def create_model(self) -> Model:
        input = Input(
            shape=(
                self.data_length,
                self.encoder_input_dim,
            )
        )

        mask = Lambda(self._create_enc_self_attention_mask)(input)

        enc_output = Encoder(
            self.hidden_dim,
            self.num_layer,
            self.num_head,
            self.ffn_dim,
            self.dropout_rate,
        )(input, mask=mask)

        output = Dense(self.output_dim, activation="sigmoid")(
            enc_output
        )

        model = Model(inputs=input, outputs=output)

        model.compile(
            optimizer=Adam(),
            loss="binary_crossentropy",
            metrics=[
                BinaryAccuracy(),
                F1,
                Precision(name="precision"),
                Recall(name="recall"),
            ],
        )

        return model

    # NOTE: <PAD>を用いない実験のためマスクを作成していない
    def _create_enc_self_attention_mask(self, input: tf.Tensor):
        batch_size, length, _ = tf.unstack(tf.shape(input))

        array = tf.zeros([batch_size, length], dtype=tf.bool)
        return tf.reshape(array, [batch_size, 1, 1, length])
