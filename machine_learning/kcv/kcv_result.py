import os
from pathlib import Path


class kcv_result:
    def __init__(self, root_dir: str, k: int) -> None:
        self.root_dir = root_dir
        self.results_dir = os.path.join(self.root_dir, "result_{}fold".format(k))
        self.figures_dir = os.path.join(self.results_dir, "figures")
        self.folds_dirs = [
            os.path.join(self.figures_dir, "figure_fold{}".format(i)) for i in range(k)
        ]
        self.average_result_dir = os.path.join(self.figures_dir, "kcv_result_average")
        self.histories_dir = os.path.join(self.results_dir, "histories")
        self.model_weights_dir = os.path.join(self.results_dir, "model_weights")

        Path(self.root_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figures_dir).mkdir(parents=True, exist_ok=True)
        map(lambda path: Path(path).mkdir(parents=True, exist_ok=True), self.folds_dirs)
        Path(self.average_result_dir).mkdir(parents=True, exist_ok=True)
        Path(self.histories_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_weights_dir).mkdir(parents=True, exist_ok=True)
        