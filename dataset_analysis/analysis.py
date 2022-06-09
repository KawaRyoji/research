from matplotlib import pyplot as plt
import pandas as pd
from musicnet.metadata import metadata
from musicnet.annotation import dataset_label
from util.fig import graph_settings
from util.path import dir2paths

d = metadata("./resource/musicnet/musicnet_metadata.csv")
d.plot_tempo_marking_bar("./dataset_analysis/slogan.png")
d.plot_ensemble_bar("./dataset_analysis/ensemble.png")

train_label_paths = dir2paths("./resource/musicnet16k/train_labels")
test_label_paths = dir2paths("./resource/musicnet16k/test_labels")

train_labels = list(map(dataset_label.load, train_label_paths))
test_labels = list(map(dataset_label.load, test_label_paths))

notes = []

for label in train_labels:
    notes.extend(label.note)

for label in test_labels:
    notes.extend(label.note)

notes_df = pd.Series(notes)
notes_df.hist(grid=True, bins=notes_df.max() - notes_df.min())
graph_settings(
    xlabel="Note number",
    ylabel="The number of notes",
    close=True,
    savefig_path="./dataset_analysis/notes.png",
)
