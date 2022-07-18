import os
from pathlib import Path


class holdout_result:
    """
    ホールドアウト法の保存先ディレクトリを定義するクラスです。
    """
    def __init__(self, root_dir: str) -> None:
        """
        Args:
            root_dir (str): ルートディレクトリのパス
        """
        self.root_dir = root_dir
        self.results_dir = os.path.join(self.root_dir, "holdout_result")
        self.figures_dir = os.path.join(self.results_dir, "figures")
        self.history_path = os.path.join(self.results_dir, "history.csv")
        self.model_weight_dir = os.path.join(self.results_dir, "model_weights")
        self.model_weight_path = os.path.join(self.model_weight_dir, "best_model_weight.ckpt")
        
        Path(self.root_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figures_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_weight_dir).mkdir(parents=True, exist_ok=True)
