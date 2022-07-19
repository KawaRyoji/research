import os
import machine_learning.comparison_plot as cplot
from machine_learning.learning_history import learning_history

lstm = "./experimental_results/transformer/lstm"
mlp = "./experimental_results/transformer/mlp"

# デコーダーがLSTMの実験

# 時間波形

lstm_waveform_w1024_norm = learning_history.from_path(
    os.path.join(lstm, "waveform/w1024_norm/holdout_result/history.csv")
)

lstm_waveform_w1024_not_norm = learning_history.from_path(
    os.path.join(lstm, "waveform/w1024_not_norm/holdout_result/history.csv")
)

# スペクトログラム

lstm_spectrogram_w1024_norm = learning_history.from_path(
    os.path.join(lstm, "spectrogram/w1024_norm/holdout_result/history.csv")
)

lstm_spectrogram_w1024_not_norm = learning_history.from_path(
    os.path.join(lstm, "spectrogram/w1024_not_norm/holdout_result/history.csv")
)

lstm_spectrogram_w2048_norm = learning_history.from_path(
    os.path.join(lstm, "spectrogram/w2048_norm/holdout_result/history.csv")
)

lstm_spectrogram_w2048_not_norm = learning_history.from_path(
    os.path.join(lstm, "spectrogram/w2048_not_norm/holdout_result/history.csv")
)

# 対数スペクトログラム

lstm_log_spectrogram_w1024_norm = learning_history.from_path(
    os.path.join(lstm, "log_spectrogram/w1024_norm/holdout_result/history.csv")
)

lstm_log_spectrogram_w1024_not_norm = learning_history.from_path(
    os.path.join(lstm, "log_spectrogram/w1024_not_norm/holdout_result/history.csv")
)

lstm_log_spectrogram_w2048_norm = learning_history.from_path(
    os.path.join(lstm, "log_spectrogram/w2048_norm/holdout_result/history.csv")
)

lstm_log_spectrogram_w2048_not_norm = learning_history.from_path(
    os.path.join(lstm, "log_spectrogram/w2048_not_norm/holdout_result/history.csv")
)


# デコーダーがMLPのみの実験

# 時間波形

mlp_waveform_w1024_norm = learning_history.from_path(
    os.path.join(mlp, "waveform/w1024_norm/holdout_result/history.csv")
)

mlp_waveform_w1024_not_norm = learning_history.from_path(
    os.path.join(mlp, "waveform/w1024_not_norm/holdout_result/history.csv")
)

# スペクトログラム

mlp_spectrogram_w1024_norm = learning_history.from_path(
    os.path.join(mlp, "spectrogram/w1024_norm/holdout_result/history.csv")
)

mlp_spectrogram_w1024_not_norm = learning_history.from_path(
    os.path.join(mlp, "spectrogram/w1024_not_norm/holdout_result/history.csv")
)

mlp_spectrogram_w2048_norm = learning_history.from_path(
    os.path.join(mlp, "spectrogram/w2048_norm/holdout_result/history.csv")
)

mlp_spectrogram_w2048_not_norm = learning_history.from_path(
    os.path.join(mlp, "spectrogram/w2048_not_norm/holdout_result/history.csv")
)

# 対数スペクトログラム

mlp_log_spectrogram_w1024_norm = learning_history.from_path(
    os.path.join(mlp, "log_spectrogram/w1024_norm/holdout_result/history.csv")
)

mlp_log_spectrogram_w1024_not_norm = learning_history.from_path(
    os.path.join(mlp, "log_spectrogram/w1024_not_norm/holdout_result/history.csv")
)

mlp_log_spectrogram_w2048_norm = learning_history.from_path(
    os.path.join(mlp, "log_spectrogram/w2048_norm/holdout_result/history.csv")
)

mlp_log_spectrogram_w2048_not_norm = learning_history.from_path(
    os.path.join(mlp, "log_spectrogram/w2048_not_norm/holdout_result/history.csv")
)


fig_dir = "./comparison_results/transformer"

# LSTM内での比較

cplot.plot_histories(
    os.path.join(fig_dir, "lstm_waveform"),
    lstm_waveform_w1024_norm,
    lstm_waveform_w1024_not_norm,
    legend=["normalized", "not normalized"],
)

cplot.plot_histories(
    os.path.join(fig_dir, "lstm_spectrogram"),
    lstm_spectrogram_w1024_norm,
    lstm_spectrogram_w1024_not_norm,
    lstm_spectrogram_w2048_norm,
    lstm_spectrogram_w2048_not_norm,
    legend=[
        "normalized(len=1024)",
        "not normalized(len=1024)",
        "normalized(len=2048)",
        "not normalized(len=2048)",
    ],
)

cplot.plot_histories(
    os.path.join(fig_dir, "lstm_log_spectrogram"),
    lstm_log_spectrogram_w1024_norm,
    lstm_log_spectrogram_w1024_not_norm,
    lstm_log_spectrogram_w2048_norm,
    lstm_log_spectrogram_w2048_not_norm,
    legend=[
        "normalized(len=1024)",
        "not normalized(len=1024)",
        "normalized(len=2048)",
        "not normalized(len=2048)",
    ],
)

# MLP内での比較

cplot.plot_histories(
    os.path.join(fig_dir, "mlp_waveform"),
    mlp_waveform_w1024_norm,
    mlp_waveform_w1024_not_norm,
    legend=["normalized", "not normalized"],
)

cplot.plot_histories(
    os.path.join(fig_dir, "mlp_spectrogram"),
    mlp_spectrogram_w1024_norm,
    mlp_spectrogram_w1024_not_norm,
    mlp_spectrogram_w2048_norm,
    mlp_spectrogram_w2048_not_norm,
    legend=[
        "normalized(len=1024)",
        "not normalized(len=1024)",
        "normalized(len=2048)",
        "not normalized(len=2048)",
    ],
)

cplot.plot_histories(
    os.path.join(fig_dir, "mlp_log_spectrogram"),
    mlp_log_spectrogram_w1024_norm,
    mlp_log_spectrogram_w1024_not_norm,
    mlp_log_spectrogram_w2048_norm,
    mlp_log_spectrogram_w2048_not_norm,
    legend=[
        "normalized(len=1024)",
        "not normalized(len=1024)",
        "normalized(len=2048)",
        "not normalized(len=2048)",
    ],
)

# LSTMとMLPの比較(testで最もLoss値が低かったもの)
cplot.plot_histories(
    os.path.join(fig_dir, "model_waveform"),
    lstm_waveform_w1024_norm,
    mlp_waveform_w1024_norm,
    legend=["LSTM", "MLP"]
)

cplot.plot_histories(
    os.path.join(fig_dir, "model_spectrogram"),
    lstm_spectrogram_w2048_not_norm,
    mlp_spectrogram_w1024_not_norm,
    legend=["LSTM", "MLP"]
)

cplot.plot_histories(
    os.path.join(fig_dir, "model_log_spectrogram"),
    lstm_log_spectrogram_w2048_not_norm,
    mlp_log_spectrogram_w1024_not_norm,
    legend=["LSTM", "MLP"]
)