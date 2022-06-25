import pandas as pd
from machine_learning.metrics import F1_from_log
from typing import List


class learning_history:
    """
    tensorflowのCSVLoggerで保存された学習履歴を扱うクラスです。
    """

    def __init__(self, df_history: pd.DataFrame, metrics: list) -> None:
        """
        Args:
            df_history (pd.DataFrame): 学習履歴のcsvを読み込んだデータフレーム
            metrics (list): 扱う評価値
        """
        self.df_history = df_history
        self.metrics = metrics

    @classmethod
    def from_path(cls, history_path: str) -> "learning_history":
        """
        学種履歴のcsvのパスを指定してインスタンスを生成します

        Args:
            history_path (str): 学習履歴のcsvのパス

        Returns:
            learning_history: 生成したインスタンス
        """
        df_history = pd.read_csv(history_path, index_col=0)
        metrics = df_history.columns.to_list()

        return cls(df_history, metrics)

    @classmethod
    def from_dir(cls, history_dir: str) -> List["learning_history"]:
        """
        学習履歴が保存されているディレクトリから、リスト形式でインスタンスを生成します。

        Returns:
            List[learning_history]: ディレクトリに含まれる学習履歴から生成したインスタンスのリスト
        """
        import util.path

        history_paths = util.path.dir2paths(history_dir)
        histories = list(
            map(lambda history_path: cls.from_path(history_path), history_paths)
        )

        return histories

    @classmethod
    def average(cls, *learning_histories: "learning_history") -> "learning_history":
        """
        複数の学習履歴から、各評価値ごとに平均を計算し、学習履歴のインスタンスを返します。

        Returns:
            learning_history: 平均された学習履歴のインスタンス
        """
        df_histories = list(map(lambda x: x.df_history, learning_histories))
        df_avg = sum(df_histories) / len(df_histories)
        metrics = df_avg.columns.to_list()

        return cls(df_avg, metrics)

    def apply_F1_from_log(self) -> "learning_history":
        """
        学習履歴からF1値を計算し、その結果を踏まえたインスタンスを返します。

        Returns:
            learning_history: F1値を学習履歴から計算したインスタンス
        """
        applied = F1_from_log(self.df_history)

        return learning_history(applied, self.metrics)

    def filter_by_metrics(self, metrics: List[str]) -> "learning_history":
        """
        指定した評価値でフィルタリングします。

        Args:
            metrics (List[str]): フィルタリングしたい評価値

        Returns:
            learning_history: 指定した評価値でフィルタリングされた学習履歴のインスタンス
        """
        filtered = learning_history(self.df_history.filter(items=metrics), metrics)

        return filtered

    def melt(self) -> pd.DataFrame:
        """
        横持ちのデータを縦持ちに変換したデータフレームを返します。
        箱ひげ図のプロットに必要となります。

        Returns:
            pd.DataFrame: 変換したデータフレーム
        """
        return pd.melt(self.df_history, ignore_index=True)

    def of_metric(self, metric: str) -> pd.DataFrame:
        """
        指定した評価値のデータフレームを返します。

        Args:
            metric (str): 指定する評価値

        Returns:
            pd.DataFrame: 指定した評価値に対応するデータフレーム
        """
        return self.df_history[metric]

    def save_to(self, file_path: str):
        """
        学習履歴を指定したパスに保存します。

        Args:
            file_path (str): 保存先のパス(csvファイル)
        """
        pd.DataFrame.to_csv(self.df_history, file_path)

    def __str__(self) -> str:
        return self.df_history.__str__()
