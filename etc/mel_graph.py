from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from machine_learning.plot import graph_settings

sns.set()  # NOTE: subplots()を呼び出す前に呼び出さないと、グリッドが一番手前に表示されてしまう
sns.set_style("white")  # グラフのスタイルを指定
sns.set_palette("Set1")  # ここでカラーパレットを変える

linear = np.arange(8000)
mel = 1127.010480 * np.log(linear / 700.0 + 1)

plt.plot(mel)

graph_settings(
    savefig_path="mel_graph.png",
    close=True,
    xlabel="frequency (Hz)",
    ylabel="mel scale(mel)",
    xlim=[0, 8000],
    ylim=[0, max(mel)]
)
