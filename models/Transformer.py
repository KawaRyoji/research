import tensorflow as tf

from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras import Model
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.layers import Input, Lambda, Dense, Layer
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
        warmup_step: int = 4000,
        decoder: Layer = None,
    ) -> None:
        self.data_length = data_length
        self.encoder_input_dim = encoder_input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.num_head = num_head
        self.dropout_rate = dropout_rate
        self.warmup_step = warmup_step
        self.decoder = decoder

    def create_model(self) -> Model:
        input = Input(
            shape=(
                self.data_length,
                self.encoder_input_dim,
            )
        )

        mask = Lambda(self._create_enc_self_attention_mask)(input)

        y = Encoder(
            self.hidden_dim,
            self.num_layer,
            self.num_head,
            self.ffn_dim,
            self.dropout_rate,
        )(input, mask=mask)

        if self.decoder is not None:
            y = self.decoder(y)
            
        output = Dense(self.output_dim, activation="sigmoid")(y)
        
        model = Model(inputs=input, outputs=output)

        scheduler = TransformerLearningRateScheduler()
        
        model.compile(
            optimizer=Adam(learning_rate=scheduler, beta_2=0.98),
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


class TransformerLearningRateScheduler(LearningRateSchedule):
    def __init__(self, max_learning_rate=0.0001, warmup_step=4000) -> None:
        self.max_learning_rate = max_learning_rate
        self.warmup_step = warmup_step

    def __call__(self, step) -> float:
        rate = tf.minimum(step ** -0.5, step * self.warmup_step ** -1.5) / self.warmup_step ** -0.5
        return self.max_learning_rate * rate
