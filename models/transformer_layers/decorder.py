import tensorflow as tf
from models.common_layers.attention import MultiHeadAttention, SelfAttention
from models.transformer_layers.embeddings import (AddPositionalEncoding,
                                                  TokenEmbedding)
from models.transformer_layers.feed_forward_network import FeedForwardNetwork
from models.transformer_layers.residual_normalization_wrapper import \
    ResidualNormalizationWrapper
from tensorflow.keras.layers import Dense, Dropout, Layer, LayerNormalization


class Decoder(Layer):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layer: int,
        num_head: int,
        ffn_dim: int,
        dropout_rate: float,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.num_head = num_head
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        self.token_embedding = TokenEmbedding(hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.dropout = Dropout(dropout_rate)

        self.dec_blocks = []
        for _ in range(num_layer):
            self_attention_layer = SelfAttention(hidden_dim, num_head, dropout_rate)
            enc_dec_attention_layer = MultiHeadAttention(
                hidden_dim, num_head, dropout_rate
            )
            ffn_layer = FeedForwardNetwork(hidden_dim, ffn_dim, dropout_rate)

            self.dec_blocks.append(
                [
                    ResidualNormalizationWrapper(self_attention_layer, dropout_rate),
                    ResidualNormalizationWrapper(enc_dec_attention_layer, dropout_rate),
                    ResidualNormalizationWrapper(ffn_layer, dropout_rate),
                ]
            )

        self.layer_normalization = LayerNormalization()
        self.dense = Dense(output_dim)

    def call(
        self,
        input: tf.Tensor,
        encoder_output: tf.Tensor,
        self_attention_mask: tf.Tensor,
        enc_dec_attention_mask: tf.Tensor,
    ):
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.dropout(embedded_input)

        for i, layers in enumerate(self.dec_blocks):
            self_attention_layer, enc_dec_attention_layer, ffn_layer = tuple(layers)

            query = self_attention_layer(query, mask=self_attention_mask)
            query = enc_dec_attention_layer(
                query, memory=encoder_output, mask=enc_dec_attention_mask
            )
            query = ffn_layer(query)

        query = self.layer_normalization(query)
        return self.dense(query)
