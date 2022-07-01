from typing import Tuple

import numpy as np
from machine_learning.dataset import data_sequence


class kcv_data_sequence:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        k: int,
        batch_size: int,
        batches_per_epoch=None,
        valid_size=None,
    ) -> None:
        self.x = x
        self.y = y
        self.k = k
        self.batch_size = batch_size
        self.valid_size = valid_size

        if batches_per_epoch is None:
            self.batches_per_epoch = len(self.x) // self.batch_size
        else:
            self.batches_per_epoch = batches_per_epoch

    def generate(
        self,
    ) -> Tuple[int, data_sequence, Tuple[np.ndarray, np.ndarray]]:
        fold_size = len(self.x) // self.k

        for fold in range(self.k):
            start = fold * fold_size
            end = start + fold_size

            mask = np.ones(len(self.x), dtype=bool)
            mask[start:end] = False

            x_valid = self.x[start:end]
            y_valid = self.y[start:end]

            x_train = self.x[mask]
            y_train = self.y[mask]

            del mask
            sequence = data_sequence(
                x_train,
                y_train,
                self.batch_size,
                batches_per_epoch=self.batches_per_epoch,
            )

            if self.valid_size is not None:
                x_valid = x_valid[: self.valid_size]
                y_valid = y_valid[: self.valid_size]

            yield (fold, sequence, (x_valid, y_valid))
