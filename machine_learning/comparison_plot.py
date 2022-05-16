import os
from matplotlib import pyplot as plt
from util.fig import graph_plot, graph_settings
import pandas as pd
import seaborn as sns
from machine_learning.learning_history import learning_history
from typing import List

sns.set()  # NOTE: subplots()を呼び出す前に呼び出さないと、グリッドが一番手前に表示されてしまう
sns.set_style("white")  # グラフのスタイルを指定
sns.set_palette("Set1")  # ここでカラーパレットを変える

# TODO: コメントの書き直し
def plot_histories(
    savefig_dir: str,
    *histories: learning_history,
    metrics: List[str] = None,
    legend: List[str] = None
):
    if legend is None:
        legend = ["history{}".format(i + 1) for i in range(len(histories))]
    else:
        if len(legend) != len(histories):
            raise "history_pathsとlegendの次元を一致させてください"

    if metrics is None:
        metrics = histories[0].metrics

    def _plot(metric: str):
        ylim = (
            min(list(map(lambda history: min(history.of_metric(metric)), histories))),
            max(list(map(lambda history: max(history.of_metric(metric)), histories))),
        )
        ylabel = metric[len("val_") :] if metric.startswith("val_") else metric
        ylabel = ylabel.replace("_", " ")

        graph_plot(
            *list(map(lambda history: history.of_metric(metric).plot, histories)),
            xlabel="epoch",
            ylabel=ylabel,
            xlim=(0, len(histories[0].of_metric(metric)) - 1),
            ylim=ylim,
            savefig_path=os.path.join(savefig_dir, metric + "_compare.png"),
            legend=legend,
            close=True,
        )

    for metric in metrics:
        _plot(metric)


def box_plot_histories(
    savefig_path: str,
    *histories: learning_history,
    stripplot=False,
    legend: List[str] = None,
    metrics: List[str] = None
):
    if legend is None:
        legend = ["history{}".format(i + 1) for i in range(len(histories))]
    else:
        if len(legend) != len(histories):
            raise "history_pathsとlegendの次元を一致させてください"

    if metrics is None:
        metrics = histories[0].metrics
        metrics = list(filter(lambda c: not c == "loss"))

    filtered_histories = list(
        map(lambda history: history.filter_by_metrics(metrics), histories)
    )

    melted_hisories = list(
        map(lambda history: history.melt(), filtered_histories)
    )
    
    for melted_history, l in zip(melted_hisories, legend):
        melted_history["group"] = l

    df = pd.concat(melted_hisories, axis=0)

    fig, ax = plt.subplots()
    sns.boxplot(x="variable", y="value", data=df, hue="group", whis=[0, 100], ax=ax)

    h, l = ax.get_legend_handles_labels()
    if stripplot:
        sns.stripplot(
            x="variable",
            y="value",
            data=df,
            hue="group",
            dodge=True,
            jitter=True,
            color="black",
            ax=ax,
        )
        h, l = ax.get_legend_handles_labels()
        h = h[: len(h) // 2]
        l = l[: len(l) // 2]

    ax.legend(h, l)
    graph_settings(
        xlabel="metrics",
        ylabel="value",
        savefig_path=savefig_path,
        close=True,
    )
