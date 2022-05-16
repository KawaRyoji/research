import keras.backend as K
import librosa
import numpy as np

BINS_PER_OCTAVE = 12
N_OCTAVES = 10
HARMONICS = [0.5, 1, 2, 3, 4, 5]
SR = 16000
FMIN = 8.2
HOP_LENGTH = 256


def compute_hcqt(audio_fpath):
    y, fs = librosa.load(audio_fpath, sr=SR)

    cqt_list = []
    shapes = []
    for h in HARMONICS:
        cqt = librosa.cqt(
            y,
            sr=fs,
            hop_length=HOP_LENGTH,
            fmin=FMIN * float(h),
            n_bins=BINS_PER_OCTAVE * N_OCTAVES,
            bins_per_octave=BINS_PER_OCTAVE,
        )
        cqt_list.append(cqt)
        shapes.append(cqt.shape)

    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[1] for s in shapes])
        new_cqt_list = []
        for i in range(len(cqt_list)):
            new_cqt_list.append(cqt_list[i][:, :min_time])
        cqt_list = new_cqt_list

    log_hcqt = (
        (1.0 / 80.0)
        * librosa.core.amplitude_to_db(np.abs(np.array(cqt_list)), ref=np.max)
    ) + 1.0

    return log_hcqt


def create_model():
    from keras.layers import Input, BatchNormalization, Conv2D, Lambda
    from keras.models import Model
    from machine_learning.metrics import F1
    from keras.metrics import Precision, Recall
    from keras.optimizers import Adam

    input_shape = (None, None, 6)
    inputs = Input(shape=input_shape)

    y0 = BatchNormalization()(inputs)
    y1 = Conv2D(128, (5, 5), padding="same", activation="relu", name="bendy1")(y0)
    y1a = BatchNormalization()(y1)
    y2 = Conv2D(64, (5, 5), padding="same", activation="relu", name="bendy2")(y1a)
    y2a = BatchNormalization()(y2)
    y3 = Conv2D(64, (3, 3), padding="same", activation="relu", name="smoothy1")(y2a)
    y3a = BatchNormalization()(y3)
    y4 = Conv2D(64, (3, 3), padding="same", activation="relu", name="smoothy2")(y3a)
    y4a = BatchNormalization()(y4)
    y5 = Conv2D(8, (70, 3), padding="same", activation="relu", name="distribute")(y4a)
    y5a = BatchNormalization()(y5)
    y6 = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name="squishy")(y5a)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y6)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(
        loss="binary_crossentropy",
        metrics=[F1, Precision(name="precision"), Recall(name="recall")],
        optimizer=Adam(),
    )
    return model
