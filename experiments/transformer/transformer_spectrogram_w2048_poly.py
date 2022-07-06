import os

from experiments.features.spectrogram import construct_process
from experiments.ho_experiment import ho_experiment
from machine_learning.dataset import dataset
from machine_learning.parameter import hyper_params
from models.Transformer import Transformer
from tensorflow.keras.callbacks import EarlyStopping

train_data_dir = "./resource/musicnet16k/train_data"
train_label_dir = "./resource/musicnet16k/train_labels"
test_data_dir = "./resource/musicnet16k/test_data"
test_label_dir = "./resource/musicnet16k/test_labels"

predict_data_path = "./resource/musicnet16k/test_data/2556.wav"
predict_label_path = "./resource/musicnet16k/test_labels/2556.csv"

experimental_result_dir = "./experimental_results/transformer_spectrogram_w2048_poly"

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

params = hyper_params(32, 1000, epoch_size=500, learning_rate=0.0001)
flen = 2048
time_len = 32
threshold = 0.5
normalize = True

model = Transformer(data_length=time_len)
ex = ho_experiment(
    model=model,
    train_set=train_set,
    test_set=test_set,
    params=params,
    experimental_result_dir=experimental_result_dir,
)

es_callback = EarlyStopping(patience=5)

ex.prepare_dataset(normalize=normalize, flen=flen, time_len=time_len)
ex.train(
    callbacks=[es_callback], valid_limit=params.batch_size * params.epoch_size // 4
)
ex.test()

prediction, labels = ex.predict(
    predict_data_path,
    predict_label_path,
    flen=flen,
    time_len=time_len,
    normalize=normalize
)

ex.plot_concat_prediction(
    prediction,
    labels,
    os.path.join(
        ex.results.figures_dir,
        "predict_" + os.path.basename(predict_data_path) + ".png",
    )
)

ex.plot_concat_prediction(
    prediction,
    labels,
    os.path.join(
        ex.results.figures_dir,
        "predict_"
        + os.path.basename(predict_data_path)
        + "_th{:.2f}.png".format(threshold),
    ),
    threshold=threshold,
)
