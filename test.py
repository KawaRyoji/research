import numpy as np
from machine_learning.dataset import dataset, data_sequence
from models.Transformer import Transformer
from tensorflow.keras.layers import LSTM, Permute
from tensorflow.keras.models import Sequential
import tensorflow as tf

# x = np.arange(1000)
# y = np.arange(1000)

# x = np.reshape(x, [10, 10, 10])
# y = np.reshape(y, [10, 10, 10])

# seq = data_sequence(x, y, 20)

# print(seq.data)
# print(seq.labels)
# seq.on_epoch_end()
# print(seq.data)
# print(seq.labels)



decoder = Sequential(
    layers=[Permute((2, 1)), LSTM(32, return_sequences=True), Permute((2, 1))],
    name="decoder",
)
model = Transformer(data_length=32, decoder=decoder)
model = model.create_model()
print(model.summary())

devices = tf.config.list_physical_devices()
print(devices)