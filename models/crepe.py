import keras

sampling_freq = 16000

def create_model(
    input_size=1024,
    first_stride=4
) -> keras.Model:
    from keras.layers import Input, Reshape, Conv2D, BatchNormalization
    from keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense
    from keras.models import Model
    from keras.metrics import Precision, BinaryAccuracy, Recall
    from machine_learning.metrics import F1

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
    strides = [(first_stride, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    x = Input(shape=(input_size,), name="input", dtype="float32")
    y = Reshape(target_shape=(input_size, 1, 1), name="input-reshape")(x)

    for l, f, w, s in zip(layers, filters, widths, strides):
        y = Conv2D(
            f, (w, 1), strides=s, padding="same", activation="relu", name="conv%d" % l
        )(y)
        y = BatchNormalization(name="conv%d-BN" % l)(y)
        y = MaxPool2D(
            pool_size=(2, 1), strides=None, padding="valid", name="conv%d-maxpool" % l
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
