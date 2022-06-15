import machine_learning.comparison_plot as cplot
from machine_learning.learning_history import learning_history

# 時間波形
crepe_waveform = learning_history.from_path(
    "./experimental_results/crepe_waveform_nolimit_poly/holdout_result/history.csv"
)
crepe_waveform_test = learning_history.from_path(
    "./experimental_results/crepe_waveform_nolimit_poly/holdout_result/test_res.csv"
)

da_waveform = learning_history.from_path(
    "./experimental_results/da_waveform_nolimit_poly/holdout_result/history.csv"
)
da_waveform_test = learning_history.from_path(
    "./experimental_results/da_waveform_nolimit_poly/holdout_result/test_res.csv"
)

# スペクトル
crepe_spec = learning_history.from_path(
    "./experimental_results/crepe_spec_nolimit_poly/holdout_result/history.csv"
)
crepe_spec_test = learning_history.from_path(
    "./experimental_results/crepe_spec_nolimit_poly/holdout_result/test_res.csv"
)

da_spec = learning_history.from_path(
    "./experimental_results/da_spec_nolimit_poly/holdout_result/history.csv"
)
da_spec_test = learning_history.from_path(
    "./experimental_results/da_spec_nolimit_poly/holdout_result/test_res.csv"
)

# 対数スペクトル
crepe_logspec = learning_history.from_path(
    "./experimental_results/crepe_logspec_nolimit_poly/holdout_result/history.csv"
)
crepe_logspec_test = learning_history.from_path(
    "./experimental_results/crepe_logspec_nolimit_poly/holdout_result/test_res.csv"
)

da_logspec = learning_history.from_path(
    "./experimental_results/da_logspec_nolimit_poly/holdout_result/history.csv"
)
da_logspec_test = learning_history.from_path(
    "./experimental_results/da_logspec_nolimit_poly/holdout_result/test_res.csv"
)


metrics = ["precision", "recall", "F1"]

# 時間波形による比較
cplot.plot_histories(
    "./comparison_results/model_waveform",
    crepe_waveform,
    da_waveform,
    metrics=metrics,
    legend=["CREPE", "DA-Net"]
)

# スペクトルによる比較
cplot.plot_histories(
    "./comparison_results/model_spec",
    crepe_spec,
    da_spec,
    metrics=metrics,
    legend=["CREPE", "DA-Net"]
)

# 対数スペクトルによる比較
cplot.plot_histories(
    "./comparison_results/model_logspec",
    crepe_logspec,
    da_logspec,
    metrics=metrics,
    legend=["CREPE", "DA-Net"]
)

