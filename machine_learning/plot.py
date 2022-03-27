from machine_learning.metrics import F1_from_log
from util.fig import graph_plot, graph_settings
from matplotlib import pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
import numpy as np
import pandas as pd
import seaborn as sns

sns.set() # NOTE: subplots()を呼び出す前に呼び出さないと、グリッドが一番手前に表示されてしまう
sns.set_style("white") # グラフのスタイルを指定
sns.set_palette("Set1") # ここでカラーパレットを変える

def plot_history(history_path: str, savefig_dir: str):
    """
    学習で得られたhistoryからグラフにプロットし、保存します

    ## Params
        - history_path (str): .csvで保存したhistoryのパス
        - savefig_dir (str): プロットしたグラフを保存するディレクトリパス
    """

    Path.mkdir(Path(savefig_dir), parents=True, exist_ok=True)
    history = pd.read_csv(history_path)

    metrics = history.columns.to_list()
    metrics = list(
        filter(lambda c: (not c.startswith("val_")) and (not c == "epoch"), metrics)
    )

    def _plot(metric):
        plot = history[[metric, "val_" + metric]].plot
        ylim = (
            min(min(history[metric]), min(history["val_" + metric])),
            max(max(history[metric]), max(history["val_" + metric])),
        )

        graph_plot(
            plot,
            xlabel="epoch",
            ylabel=metric,
            xlim=(0, len(history) - 1),
            ylim=ylim,
            savefig_path=os.path.join(savefig_dir, metric + ".png"),
            legend=["training", "validation"],
            close=True,
        )

    for metric in metrics:
        _plot(metric)


def plot_activation(activation, savefig_path: str, gray_scale=False):
    """
    mode.predict(x)で得られた推定結果をカラーマップで保存します

    ## Params
        - activation (array): 得られた推定結果の配列
            - 内部で上下を反転する処理が入ります
        - savefig_path (str): カラーマップを保存するパス
        - gray_scale (bool): Trueの場合グレースケールで保存します.
    """

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.ioff()
    if gray_scale:
        plt.gray()
    else:
        plt.set_cmap("magma")

    Path(savefig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(savefig_path, np.flipud(activation))
    plt.close()


def plot_activation_with_labels(labels, activation, savefig_path):
    activation = activation.T
    labels = labels.T
    fig, ax = plt.subplots(
        dpi=100, figsize=(len(activation[0]) / 100, len(activation) / 100)
    )

    Path(savefig_path).parent.mkdir(parents=True, exist_ok=True)

    # 活性化マップとラベルを重ねってプロットする部分
    ax.pcolor(activation, cmap="magma")
    # hotvectorの値が0のときは透明に, 値が1のときはシアン(alpha=0.5)に
    cmap = mpl.colors.ListedColormap([(0, 0, 0, 0), (0, 1, 1, 0.5)])
    ax.pcolor(labels, cmap=cmap)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.savefig(savefig_path)
    plt.close()


def plot_compare_history(savefig_dir: str, *history_paths: str, legend: list = None):
    """
    学習から得られたhistoryを比較してプロットします

    metricsはhistory_pathsで最初に指定したhistoryに依存します

    ## Params
        - savefig_dir (str): プロットしたグラフを保存するディレクトリパス
        - history_paths (tuple[str, ...]): プロットするhistoryのパス
        - legend (list, optional): プロットするhistoryの凡例
            - Noneが指定されている場合、"history1", "history2" ... のようになります
            - 指定する場合はhistory_pathsと次元を一致させてください
    """
    if legend is None:
        legend = ["history{}".format(i + 1) for i in range(len(history_paths))]
    else:
        if len(legend) != len(history_paths):
            raise "history_pathsとlegendの次元を一致させてください"

    Path.mkdir(Path(savefig_dir), parents=True, exist_ok=True)

    df_histories = [pd.read_csv(history_path) for history_path in history_paths]

    metrics = df_histories[0].columns.tolist()
    metrics = list(filter(lambda c: not c == "epoch", metrics))

    def _plot(metric):
        ylim = (
            min([min(history[metric]) for history in df_histories]),
            max([max(history[metric]) for history in df_histories]),
        )

        ylabel = metric[len("val_") :] if metric.startswith("val_") else metric
        ylabel = ylabel.replace("_", " ")
        graph_plot(
            *[history[metric].plot for history in df_histories],
            xlabel="epoch",
            ylabel=ylabel,
            xlim=(0, len(df_histories[0]) - 1),
            ylim=ylim,
            savefig_path=os.path.join(savefig_dir, metric + "_compare.png"),
            legend=legend,
            close=True,
        )

    for metric in metrics:
        _plot(metric)


def plot_average_history(savefig_dir: str, history_dir: str, metrics=None):
    df_histories = [
        pd.read_csv(os.path.join(history_dir, history_path), index_col=0)
        for history_path in os.listdir(history_dir)
    ]

    Path.mkdir(Path(savefig_dir), parents=True, exist_ok=True)

    if metrics is None:
        metrics = df_histories[0].columns.tolist()
        metrics = list(
            filter(lambda c: (not c.startswith("val_")) and (not c == "epoch"), metrics)
        )

    df_avg = sum(df_histories) / len(df_histories)

    df_avg = F1_from_log(df_avg)

    def _plot(metric):
        ylim = (
            min(min(df_avg[metric]), min(df_avg["val_" + metric])),
            max(max(df_avg[metric]), max(df_avg["val_" + metric])),
        )
        plot = df_avg[[metric, "val_" + metric]].plot
        ylabel = metric[len("val_") :] if metric.startswith("val_") else metric
        ylabel = ylabel.replace("_", " ")
        graph_plot(
            plot,
            xlabel="epoch",
            ylabel=ylabel,
            xlim=(0, len(df_histories[0]) - 1),
            ylim=ylim,
            savefig_path=os.path.join(savefig_dir, "average_" + metric + ".png"),
            legend=["training", "validation"],
            close=True,
        )
    
    for metric in metrics:
        _plot(metric)


def plot_compare_average_history(
    savefig_dir: str, *history_dir_paths: str, legend: list = None, metrics=None
):
    if legend is None:
        legend = ["history{}".format(i + 1) for i in range(len(history_dir_paths))]
    else:
        if len(legend) != len(history_dir_paths):
            raise "history_pathsとlegendの次元を一致させてください"

    df_avgs = []
    for history_dir in history_dir_paths:
        df_histories = [
            pd.read_csv(os.path.join(history_dir, history_path), index_col=0)
            for history_path in os.listdir(history_dir)
        ]
        df_avg = sum(df_histories) / len(df_histories)
        F1_from_log(df_avg)
        df_avgs.append(df_avg)

    if metrics is None:
        metrics = df_avgs[0].columns.tolist()
        metrics = list(filter(lambda c: not c == "epoch" or not c == "loss"))

    def _plot(metric):
        ylim = (
            min([min(history[metric]) for history in df_avgs]),
            max([max(history[metric]) for history in df_avgs]),
        )

        ylabel = metric[len("val_") :] if metric.startswith("val_") else metric
        ylabel = ylabel.replace("_", " ")
        graph_plot(
            *[history[metric].plot for history in df_avgs],
            xlabel="epoch",
            ylabel=ylabel,
            xlim=(0, len(df_avgs[0]) - 1),
            ylim=ylim,
            savefig_path=os.path.join(savefig_dir, metric + "_compare.png"),
            legend=legend,
            close=True,
        )

    for metric in metrics:
        _plot(metric)


def box_plot_history(
    savefig_path: str, history_path: str, stripplot=False, metrics=None
):
    df_history = pd.read_csv(history_path, index_col=0)

    if metrics is None:
        metrics = df_history.columns.tolist()
        metrics = list(filter(lambda c: not c == "epoch" or not c == "loss"))

    df_melt = pd.melt(df_history.filter(items=metrics), ignore_index=True)

    fig, ax = plt.subplots()
    sns.boxplot(x="variable", y="value", data=df_melt, whis=[0, 100], ax=ax)

    if stripplot:
        sns.stripplot(
            x="variable", y="value", data=df_melt, jitter=True, color="black", ax=ax
        )

    graph_settings(
        xlabel="metrics", ylabel="value", savefig_path=savefig_path, close=True
    )


def box_plot_history_compare(
    savefig_path: str,
    *history_paths: str,
    stripplot=False,
    legend: list = None,
    metrics=None
):
    if legend is None:
        legend = ["history{}".format(i + 1) for i in range(len(history_paths))]
    else:
        if len(legend) != len(history_paths):
            raise "history_pathsとlegendの次元を一致させてください"

    df_histories = [
        pd.read_csv(history_path, index_col=0) for history_path in history_paths
    ]

    if metrics is None:
        metrics = df_histories[0].columns.tolist()
        metrics = list(filter(lambda c: not c == "epoch" or not c == "loss"))

    df_melts = [
        pd.melt(df_history.filter(items=metrics), ignore_index=True)
        for df_history in df_histories
    ]

    for df_melt, l in zip(df_melts, legend):
        df_melt["group"] = l

    df = pd.concat(df_melts, axis=0)
    
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
