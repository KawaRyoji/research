import math
import tensorflow as tf
from machine_learning.metrics import F1
from machine_learning.model import learning_model

from tensorflow.keras.layers import Input, Dropout, Add, Dense, Lambda, LayerNormalization
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from models.common_layers.attention import MultiHeadAttention, SelfAttention


class Transformer(learning_model):
    def __init__(
        self,
        data_length: int = 16,
        encoder_input_dim: int = 1024,
        decoder_input_dim: int = 128,
        hidden_dim: int = 512,
        ffn_dim: int = 2048,
        output_dim: int = 128,
        num_layer: int = 6,
        num_head: int = 8,
        dropout_rate: float = 0.1,
    ) -> None:
        self.data_length = data_length
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.num_head = num_head
        self.dropout_rate = dropout_rate

    def create_model(self) -> Model:
        encoder_input = Input(
            shape=(
                self.data_length,
                self.encoder_input_dim,
            ),
            dtype="float32",
            name="encoder_input",
        )

        decoder_input = Input(
            shape=(
                self.data_length + 1,  # START_TOKEN 分増える
                self.decoder_input_dim,
            ),
            dtype="float32",
            name="decoder_input",
        )

        enc_mask = Lambda(self._create_enc_self_attention_mask)(encoder_input)
        dec_mask = Lambda(self._create_dec_self_attention_mask)(decoder_input)

        encoder_output = self.encoder(encoder_input, enc_mask)
        decoder_ouptut = self.decoder(decoder_input, encoder_output, dec_mask, enc_mask)

        model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_ouptut)

        model.compile(
            optimizer=Adam(),
            loss="binary_crossentropy",  # NOTE: <PAD>を使わない実験のため通常の2元クロスエントロピーを用いる
            metrics=[
                BinaryAccuracy(),
                F1,
                Precision(name="precision"),
                Recall(name="recall"),
            ],
        )

        return model

    def encoder(self, input: tf.Tensor, mask: tf.Tensor):
        embedded_input = self.token_embedding(input)
        embedded_input = Lambda(self.add_positional_encoding)(embedded_input)
        query = Dropout(self.dropout_rate)(embedded_input)

        for i in range(self.num_layer):
            attention = SelfAttention(self.hidden_dim, self.num_head, self.dropout_rate)
            query = self.residual_normalization_wrapper(attention, query, mask=mask)
            query = self.residual_normalization_wrapper(
                self.feed_forward_network, query
            )

        return LayerNormalization()(query)

    def decoder(
        self,
        input: tf.Tensor,
        encoder_output: tf.Tensor,
        self_attention_mask: tf.Tensor,
        enc_dec_attention_mask: tf.Tensor,
    ):
        embedded_input = self.token_embedding(input)
        embedded_input = Lambda(self.add_positional_encoding)(embedded_input)
        query = Dropout(self.dropout_rate)(embedded_input)

        for i in range(self.num_layer):
            attention = SelfAttention(
                self.hidden_dim,
                self.num_head,
                self.dropout_rate,
            )
            enc_dec_attention = MultiHeadAttention(
                self.hidden_dim,
                self.num_head,
                self.dropout_rate,
            )

            query = self.residual_normalization_wrapper(
                attention, query, mask=self_attention_mask
            )
            query = self.residual_normalization_wrapper(
                enc_dec_attention,
                query,
                memory=encoder_output,
                mask=enc_dec_attention_mask,
            )
            query = self.residual_normalization_wrapper(
                self.feed_forward_network, query
            )

        query = LayerNormalization()(query)

        query = Lambda(
            lambda x: x[:, 1:, :],
            output_shape=lambda shape: (shape[0], shape[1] - 1, shape[2]),
        )(
            query
        )  # START_TOKENの分を切り捨てる
        return Dense(self.output_dim, activation="sigmoid")(query)

    def feed_forward_network(self, input: tf.Tensor) -> tf.Tensor:
        output = Dense(self.ffn_dim, activation="relu")(input)
        output = Dropout(self.dropout_rate)(output)
        return Dense(self.hidden_dim)(output)

    def residual_normalization_wrapper(
        self, layer, input: tf.Tensor, *args, **kwargs
    ) -> tf.Tensor:
        output = LayerNormalization()(input)
        output = layer(output, *args, **kwargs)
        output = Dropout(self.dropout_rate)(output)

        return Add()([input, output])

    # NOTE: NLPではトークンidからembeddingに変換する処理
    # ここでは次元をhidden_dimに落とすために全結合層を用いる
    def token_embedding(self, input: tf.Tensor) -> tf.Tensor:
        return Dense(self.hidden_dim, use_bias=False)(input)

    def add_positional_encoding(self, input: tf.Tensor) -> tf.Tensor:
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

    # NOTE: <PAD>を用いない実験のためマスクを作成していない
    def _create_enc_self_attention_mask(self, input: tf.Tensor):
        batch_size, length, _ = tf.unstack(tf.shape(input))

        array = tf.zeros([batch_size, length], dtype=tf.bool)
        return tf.reshape(array, [batch_size, 1, 1, length])

    def _create_dec_self_attention_mask(self, input: tf.Tensor):
        batch_size, length, _ = tf.unstack(tf.shape(input))

        # NOTE: <PAD>を用いない実験のためマスクを作成していない
        array = tf.zeros([batch_size, length], dtype=tf.bool)
        array = tf.reshape(array, [batch_size, 1, 1, length])

        autoregression_array = tf.logical_not(
            tf.linalg.band_part(tf.ones([length, length], dtype=tf.bool), -1, 0)
        )
        autoregression_array = tf.reshape(autoregression_array, [1, 1, length, length])

        # Trueのところがマスクされる部分
        return tf.logical_or(array, autoregression_array)
