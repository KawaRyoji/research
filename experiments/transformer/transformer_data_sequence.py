from tensorflow.keras.utils import Sequence

from machine_learning.dataset import dataset


class transformer_data_sequence(Sequence):
    def __init__(self, data, labels, batch_size, batches_per_epoch=None) -> None:
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        if batches_per_epoch is None:
            self.batches_per_epoch = len(self.data) // self.batch_size
        else:
            self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        # START_TOKENを削除
        labels = self.labels[:, 1:, :]
        return {
            "encoder_input": self.data[start:end],
            "decoder_input": self.labels[start:end],
        }, labels[start:end]

    def on_epoch_end(self):
        dataset.shuffle_data(self.data, self.labels)
