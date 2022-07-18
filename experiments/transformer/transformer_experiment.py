import os
import tensorflow as tf

import experiments.features.log_spectrogram as log_spec
import experiments.features.spectrogram as spec
import experiments.features.waveform_image as waveform
from experiments.ho_experiment import ho_experiment
from experiments.kcv_experiment import kcv_experiment
from machine_learning.dataset import dataset
from machine_learning.parameter import hyper_params
from models.Transformer import Transformer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session

train_data_dir = "./resource/musicnet16k/train_data"
train_label_dir = "./resource/musicnet16k/train_labels"
test_data_dir = "./resource/musicnet16k/test_data"
test_label_dir = "./resource/musicnet16k/test_labels"

predict_data_path = "./resource/musicnet16k/test_data/2556.wav"
predict_label_path = "./resource/musicnet16k/test_labels/2556.csv"

# NOTE 重要: このコードはdnn環境(Tensorflow2.1)では正しく動作しない
# Tensorflow2.6では動作確認済み

class TransformerExperiment:
    def __init__(
        self,
        output_root_dir: str,
        feature_name: str,
        params: hyper_params,
        decoder=None,
        flen=1024,
        time_len=32,
        threshold=0.5,
        normalize=False,
        holdout=True,
        k=None,
        gpu=0,
    ) -> None:
        self.process = self._define_process(feature_name)
        self.flen = flen
        self.time_len = time_len
        self.threshold = threshold
        self.normalize = normalize
        self.params = params
        self.model = Transformer(data_length=time_len, decoder=decoder)
        self.save_dir_path = self._define_folder_name(output_root_dir, feature_name)

        train_set = dataset.from_dir(train_data_dir, train_label_dir, self.process)
        test_set = dataset.from_dir(test_data_dir, test_label_dir, self.process)

        if holdout:
            self.experiment = ho_experiment(
                self.model, train_set, test_set, params, self.save_dir_path
            )
        else:
            if k is None:
                raise RuntimeError("k must not be None")
            self.experiment = kcv_experiment(
                self.model, train_set, test_set, params, k, self.save_dir_path
            )

        try:
            physical_devices = tf.config.list_physical_devices("GPU")
            tf.config.set_visible_devices(physical_devices[gpu], "GPU")
            tf.config.experimental.set_memory_growth(physical_devices[gpu], True)
        except:
            pass

    def _define_folder_name(self, root_dir: str, feature_name: str):
        is_normalized = "norm" if self.normalize else "not_norm"
        save_dir_path = os.path.join(
            root_dir, feature_name, "w{}_".format(self.flen) + is_normalized
        )

        return save_dir_path

    def _define_process(self, feature_name: str):
        if feature_name == "spectrogram":
            return spec.construct_process
        elif feature_name == "log_spectrogram":
            return log_spec.construct_process
        elif feature_name == "waveform":
            return waveform.construct_process
        else:
            raise RuntimeError("feature name error")

    def run(self):
        self.prepare_datasets()
        clear_session()
        self.train()
        self.test()
        self.predict()

    def prepare_datasets(self):
        self.experiment.prepare_dataset(
            normalize=self.normalize, flen=self.flen, time_len=self.time_len
        )

    def train(self, early_stopping=True):
        callbacks = []

        if early_stopping:
            callbacks.append(EarlyStopping(monitor="val_F1", mode="max", patience=5))

        self.experiment.train(
            callbacks=callbacks,
            valid_limit=self.params.batch_size * self.params.epochs // 4,
        )

    def test(self):
        self.experiment.test()

    def predict(self):
        prediction, labels = self.experiment.predict(
            predict_data_path,
            predict_label_path,
            normalize=self.normalize,
            flen=self.flen,
            time_len=self.time_len,
        )

        self.experiment.plot_concat_prediction(
            prediction,
            labels,
            os.path.join(
                self.experiment.results.figures_dir,
                "predict_" + os.path.basename(predict_data_path) + ".png",
            ),
        )

        self.experiment.plot_concat_prediction(
            prediction,
            labels,
            os.path.join(
                self.experiment.results.figures_dir,
                "predict_"
                + os.path.basename(predict_data_path)
                + "_th{:.2f}.png".format(self.threshold),
            ),
            threshold=self.threshold,
        )
