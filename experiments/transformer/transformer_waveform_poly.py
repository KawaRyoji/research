import os

import tensorflow.keras.backend as K
from experiments.features.transformer_waveform import construct_process
from experiments.transformer.transformer_data_sequence import \
    transformer_data_sequence
from machine_learning.dataset import dataset
from machine_learning.holdout.holdout_result import holdout_result
from machine_learning.learning_history import learning_history
from machine_learning.model import learning_model
from machine_learning.parameter import hyper_params
from machine_learning.plot import plot_history
from models.Transformer import Transformer
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint


def train(
    learning_model: learning_model,
    params: hyper_params,
    x,
    y,
    valid_split,
    result: holdout_result,
):
    model = learning_model.create_model()

    if not model._is_compiled:
        print("Model should be compiled.")
        return

    K.set_value(model.optimizer.lr, params.learning_rate)

    train_x, train_y, valid_x, valid_y = dataset.split_data(x, y, valid_split)
    params.save_to_json(os.path.join(result.results_dir, "params.json"))

    callbacks = []
    cp_callback = ModelCheckpoint(
        result.model_weight_path,
        monitor="val_loss",
        mode="auto",
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    )
    lg_callback = CSVLogger(result.history_path)
    callbacks.append(cp_callback)
    callbacks.append(lg_callback)

    data_sequence = transformer_data_sequence(train_x, train_y, params.batch_size)

    model.fit(
        x=data_sequence,
        epochs=params.epochs,
        callbacks=callbacks,
        validation_data=({"encoder_input": valid_x, "decoder_input": valid_y}, valid_y),
    )


def test(learning_model: learning_model, x, y, result: holdout_result):
    model = learning_model.create_model()


train_data_dir = "./resource/musicnet16k/train_data"
train_label_dir = "./resource/musicnet16k/train_labels"
test_data_dir = "./resource/musicnet16k/test_data"
test_label_dir = "./resource/musicnet16k/test_labels"

experimental_result_dir = "./experimental_results/transformer_waveform_poly"
params = hyper_params(32, 32, epoch_size=500, learning_rate=0.0001)

train_set = dataset.from_dir(train_data_dir, train_label_dir, construct_process)
test_set = dataset.from_dir(train_data_dir, train_label_dir, construct_process)

model = Transformer()
results = holdout_result(experimental_result_dir)

train_set.construct(os.path.join(experimental_result_dir, "train"))
test_set.construct(os.path.join(experimental_result_dir, "test"))

x, y = train_set.load(os.path.join(experimental_result_dir, "train"))
train(model, params, x, y, 0.25, results)

history = learning_history.from_path(results.history_path)
plot_history(history, results.figures_dir)

# x, y = test_set.load(os.path.join(experimental_result_dir, "test"), shuffle=False)
# test(model, x, y, results)
