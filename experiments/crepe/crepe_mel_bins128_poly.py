import keras
from experiments.features.mel_bins128 import construct_process
from experiments.kcv_experiment import kcv_experiment
from machine_learning.dataset import dataset
from machine_learning.model import learning_model
from machine_learning.parameter import hyper_params

class CREPE_128(learning_model):
    def create_model() -> keras.Model:
        from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                                Flatten, Input, MaxPool2D, Permute, Reshape)
        from keras.metrics import BinaryAccuracy, Precision, Recall
        from keras.models import Model
        from machine_learning.metrics import F1

        input_size = 128
        capacity_multiplier = 32
        layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        widths = [
            input_size // 2,
            input_size // 16,
            input_size // 16,
            input_size // 16,
            input_size // 16,
            input_size // 16,
        ]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        x = Input(shape=(input_size,), name="input", dtype="float32")
        y = Reshape(target_shape=(input_size, 1, 1), name="input-reshape")(x)

        for l, f, w, s in zip(layers, filters, widths, strides):
            y = Conv2D(
                f, (w, 1), strides=s, padding="same", activation="relu", name="conv%d" % l
            )(y)
            y = BatchNormalization(name="conv%d-BN" % l)(y)
            y = MaxPool2D(
                pool_size=(min(y.shape[1], 2), 1),
                strides=None,
                padding="valid",
                name="conv%d-maxpool" % l,
            )(y)
            y = Dropout(0.25, name="conv%d-dropout" % l)(y)

        y = Permute((2, 1, 3), name="transpose")(y)
        y = Flatten(name="flatten")(y)
        y = Dense(128, activation="sigmoid", name="classifier")(y)

        model = Model(inputs=x, outputs=y)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="binary_crossentropy",
            metrics=[
                BinaryAccuracy(),
                F1,
                Precision(name="precision"),
                Recall(name="recall"),
            ],
        )
        return model


train_data_dir = "./resource/musicnet16k/train_data"
train_label_dir = "./resource/musicnet16k/train_labels"
test_data_dir = "./resource/musicnet16k/test_data"
test_label_dir = "./resource/musicnet16k/test_labels"

train_set = dataset.from_dir(
    train_data_dir,
    train_label_dir,
    construct_process,
)

test_set = dataset.from_dir(
    test_data_dir,
    test_label_dir,
    construct_process,
)

experimental_result_dir = "./experimental_results/crepe_mel_bins128_poly"
params = hyper_params(32, 32, epoch_size=500, learning_rate=0.0001)


model = CREPE_128()
ex = kcv_experiment(
    model=model,
    train_set=train_set,
    test_set=test_set,
    params=params,
    k=5,
    experimantal_result_dir=experimental_result_dir,
)

ex.prepare_dataset()
ex.train()
ex.test()
