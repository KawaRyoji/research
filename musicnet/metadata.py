import pandas as pd
import matplotlib.pyplot as plt
from util.fig import graph_settings

SPEED_SLOGANS = (
    "prestissimo",
    "presto",
    "vivace",
    "allegro",
    "allegretto",
    "moderato",
    "andante",
    "larghetto",
    "adagio",
    "lento",
    "largo",
)

ENSEMBLES = (
    "solo",
    "duet",
    "trio",
    "quartet",
    "quintet",
    "sextet",
    "septet",
    "octet",
    "nonet",
    "dectet",
)

class metadata:
    def __init__(self, metadata_csv_path: str) -> None:
        self.data = pd.read_csv(metadata_csv_path)

    def plot_slogan_bar(self, savefig_path: str) -> None:
        movement = self.data["movement"]
        movement = movement.map(self._search_slogan)
        movement.value_counts().plot(kind="bar")

        plt.xticks(rotation=90, fontsize=14)
        graph_settings(
            ylabel="The number of music",
            xlabel="Speed slogan",
            savefig_path=savefig_path,
            close=True,
        )

    def _search_slogan(self, item: str):
        item = item.lower()
        for slogan in SPEED_SLOGANS:
            if slogan in item:
                return slogan

    def plot_ensemble_bar(self, savefig_path: str):
        ensemble = self.data["ensemble"]
        ensemble = ensemble.map(self._search_ensemble)
        ensemble.value_counts().plot(kind="bar")

        plt.xticks(rotation=90, fontsize=14)
        graph_settings(
            ylabel="The number of music",
            xlabel="Composition",
            savefig_path=savefig_path,
            close=True,
        )

    def _search_ensemble(self, item: str):
        item = item.lower()
        for ensemble in ENSEMBLES:
            if ensemble in item:
                return ensemble
            