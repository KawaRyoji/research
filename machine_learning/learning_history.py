import pandas as pd
from machine_learning.metrics import F1_from_log
from typing import List


class learning_history:
    def __init__(self, df_history: pd.DataFrame, metrics: list) -> None:
        self.df_history = df_history
        self.metrics = metrics

    @classmethod
    def from_path(cls, history_path: str) -> "learning_history":
        df_history = pd.read_csv(history_path, index_col=0)
        metrics = df_history.columns.to_list()

        return cls(df_history, metrics)

    @classmethod
    def from_dir(cls, history_dir: str) -> List["learning_history"]:
        import util.path

        history_paths = util.path.dir2paths(history_dir)
        histories = list(
            map(lambda history_path: cls.from_path(history_path), history_paths)
        )

        return histories

    @classmethod
    def average(cls, *learning_histories: "learning_history") -> "learning_history":
        df_histories = list(map(lambda x: x.df_history, learning_histories))
        df_avg = sum(df_histories) / len(df_histories)
        metrics = df_avg.columns.to_list()

        return cls(df_avg, metrics)

    def apply_F1_from_log(self) -> "learning_history":
        applied = F1_from_log(self.df_history)

        return learning_history(applied, self.metrics)

    def filter_by_metrics(self, metrics: List[str]) -> "learning_history":
        filtered = learning_history(self.df_history.filter(items=metrics), metrics)

        return filtered

    def melt(self) -> pd.DataFrame:
        return pd.melt(self.df_history, ignore_index=True)

    def of_metric(self, metric: str) -> pd.DataFrame:
        return self.df_history[metric]

    def save_to(self, file_path: str):
        pd.DataFrame.to_csv(self.df_history, file_path)

    def __str__(self) -> str:
        return self.df_history.__str__()
