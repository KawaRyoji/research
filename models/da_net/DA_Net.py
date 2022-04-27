import keras

fs = 16000


def create_model(input_size=1024) -> keras.Model:
    from keras.layers import Input, Reshape, Flatten, Dense
    from keras.metrics import Precision, BinaryAccuracy, Recall
    from machine_learning.metrics import F1

    layers = [1, 2, 3, 4, 5, 6]
    kernel_size = [512, 64, 64, 64, 64, 64]
    filters = [1024, 128, 128, 128, 256, 512]
    strides = [4, 1, 1, 1, 1, 1]
    pooling_size = 2

    x = Input(shape=(input_size,), name="input", dtype="float32")
    y = Reshape(target_shape=(input_size, 1), name="input-reshape")(x)

    for l, f, k, s in zip(layers, filters, kernel_size, strides):
        y = da_module(y, l, f, k, s, pooling_size)

    y = Flatten(name="flatten")(y)
    y = Dense(128, activation="sigmoid", name="classifier")(y)

    model = keras.Model(inputs=x, outputs=y)
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


def da_module(x, layer, filter, kernel_size, stride, pooling_size) -> keras.Model:
    from keras.layers import (
        Conv1D,
        BatchNormalization,
        Multiply,
        GlobalAveragePooling1D,
        Dense,
        Add,
        MaxPooling1D,
        Dropout
    )
    r = 16

    # Element-wise Attention
    y_ew_u = Conv1D(
        filter,
        kernel_size,
        strides=stride,
        padding="same",
        name="da-u-conv%d" % layer,
    )(x)
    y_ew_b = Conv1D(
        filter,
        kernel_size,
        strides=stride,
        padding="same",
        activation="sigmoid",
        name="da-b-conv%d" % layer,
    )(x)
    y_ew = Multiply(name="da-ew-mul%d" % layer)([y_ew_u, y_ew_b])

    # Channel-wise Attention
    y_cw = GlobalAveragePooling1D(name="da-cw-gap%d" % layer)(y_ew_u)
    y_cw = Dense(filter//r, activation="relu", name="da-cw-dr%d" % layer)(y_cw)
    y_cw = Dense(filter, activation="sigmoid", name="da-cw-ds%d" % layer)(y_cw)
    y_cw = Multiply(name="da-cw-mul%d" % layer)([y_ew_u, y_cw])
    
    y = Add(name="da-add%d" % layer)([y_ew, y_cw])
    y = BatchNormalization(name="da-BN%d" % layer)(y)
    y = MaxPooling1D(pool_size=pooling_size, name="da-mp%d" % layer)(y)
    y = Dropout(0.25, name="da-do%d" % layer)(y)

    return y
