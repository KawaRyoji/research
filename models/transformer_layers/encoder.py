import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer, LayerNormalization
from models.common_layers.attention import SelfAttention

from models.transformer_layers.embeddings import AddPositionalEncoding, TokenEmbedding
from models.transformer_layers.feed_forward_network import FeedForwardNetwork
from models.transformer_layers.residual_normalization_wrapper import (
    ResidualNormalizationWrapper,
)


class Encoder(Layer):
    def __init__(
        self,
        hidden_dim: int,
        num_layer: int,
        num_head: int,
        ffn_dim: int,
        dropout_rate: float,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.num_head = num_head
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        self.token_embedding = TokenEmbedding(hidden_dim)
        self.add_position_encoding = AddPositionalEncoding()
        self.dropout = Dropout(dropout_rate)

        self.enc_blocks = []
        for _ in range(num_layer):
            attention_layer = SelfAttention(hidden_dim, num_head, dropout_rate)
            ffn_layer = FeedForwardNetwork(hidden_dim, ffn_dim, dropout_rate)
            self.enc_blocks.append(
                [
                    ResidualNormalizationWrapper(attention_layer, dropout_rate),
                    ResidualNormalizationWrapper(ffn_layer, dropout_rate),
                ]
            )

        self.layer_normalization = LayerNormalization()

    def call(self, input: tf.Tensor, mask: tf.Tensor):
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_encoding(embedded_input)
        query = self.dropout(embedded_input)

        for i, layers in enumerate(self.enc_blocks):
            attention_layer, ffn_layer = tuple(layers)
            
            query = attention_layer(query, mask=mask)
            query = ffn_layer(query)

        return self.layer_normalization(query)
