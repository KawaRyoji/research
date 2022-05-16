from util.fig import graph_plot, graph_settings
from matplotlib import pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
import numpy as np
import seaborn as sns
from machine_learning.learning_history import learning_history
from typing import List

sns.set()  # NOTE: subplots()を呼び出す前に呼び出さないと、グリッドが一番手前に表示されてしまう
sns.set_style("white")  # グラフのスタイルを指定
sns.set_palette("Set1")  # ここでカラーパレットを変える

# TODO: コメントの書き直し
def plot_history(
    history: learning_history, savefig_dir: str, metrics: List[str] = None
):
    f1_applied = history.apply_F1_from_log()

    if metrics is None:
        metrics = list(filter(lambda c: (not c.startswith("val_")), f1_applied.metrics))

    def _plot(metric):
        history_train = f1_applied.of_metric(metric)
        history_valid = f1_applied.of_metric("val_" + metric)

        ylim = (
            min(
                min(history_train),
                min(history_valid),
            ),
            max(
                max(history_train),
                max(history_valid),
            ),
        )

        graph_plot(
            history_train.plot,
            history_valid.plot,
            xlabel="epoch",
            ylabel=metric,
            xlim=(0, len(history_train) - 1),
            ylim=ylim,
            savefig_path=os.path.join(savefig_dir, metric + ".png"),
            legend=["training", "validation"],
            close=True,
        )

    for metric in metrics:
        _plot(metric)


def box_plot_history(
    history: learning_history, savefig_path: str, stripplot=False, metrics=None
):
    if metrics is None:
        metrics = list(filter(lambda c: not c == "loss", history.metrics))

    filtered = history.filter_by_metrics(metrics)
    melted = filtered.melt()

    fig, ax = plt.subplots()
    sns.boxplot(x="variable", y="value", data=melted, whis=[0, 100], ax=ax)

    if stripplot:
        sns.stripplot(
            x="variable", y="value", data=melted, jitter=True, color="black", ax=ax
        )

    graph_settings(
        xlabel="metrics", ylabel="value", savefig_path=savefig_path, close=True
    )


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
