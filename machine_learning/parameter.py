import keras
from pathlib import Path
import json

# TODO: オプティマイザをモデルに適用する
class hyper_params:
    def __init__(
        self,
        batch_size,
        epochs,
        epoch_size=None,
        learning_rate=None,
        optimizer=keras.optimizers.Adam,
    ) -> None:
        if learning_rate is None:
            self.learning_rate = 0.0001
        else:
            self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.epochs = epochs
        self.optimizer = optimizer(learning_rate=learning_rate)

    def __str__(self) -> str:
        return "batch_size:{}\nepochs:{}\nepoch_size:{}\nlearning_rate:{}\noptimizer:{}".format(
            self.batch_size,
            self.epochs,
            self.epoch_size,
            self.learning_rate,
            self.optimizer.__class__.__name__,
        )

    def save_to_json(self, save_path):
        Path.mkdir(Path(save_path).parent, parents=True, exist_ok=True)

        with open(save_path, mode="w") as f:
            json.dump(
                {
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "epoch_size": self.epoch_size,
                    "learning_rate": self.learning_rate,
                    "optimizer": self.optimizer.__class__.__name__,
                },
                f,
            )
