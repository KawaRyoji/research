import machine_learning.comparison_plot as cplot
from machine_learning.learning_history import learning_history

# 時間波形　複数音高推定
waveform_poly = learning_history.from_dir(
    "./experimental_results/da_waveform_poly/result_5fold/histories"
)
waveform_poly = learning_history.average(*waveform_poly)
waveform_test_poly = learning_history.from_path(
    "./experimental_results/da_waveform_poly/result_5fold/test_res.csv"
)

# 時間波形　単音高推定
waveform_mono = learning_history.from_dir(
    "./experimental_results/da_waveform_mono/result_5fold/histories"
)
waveform_mono = learning_history.average(*waveform_mono)
waveform_test_mono = learning_history.from_path(
    "./experimental_results/da_waveform_mono/result_5fold/test_res.csv"
)

# スペクトル　複数音高推定
spec_poly = learning_history.from_dir(
    "./experimental_results/da_spec_poly/result_5fold/histories"
)
spec_poly = learning_history.average(*spec_poly)
spec_test_poly = learning_history.from_path(
    "./experimental_results/da_spec_poly/result_5fold/test_res.csv"
)

# スペクトル　単音高推定
spec_mono = learning_history.from_dir(
    "./experimental_results/da_spec_mono/result_5fold/histories"
)
spec_mono = learning_history.average(*spec_mono)
spec_test_mono = learning_history.from_path(
    "./experimental_results/da_spec_mono/result_5fold/test_res.csv"
)

# スペクトル(標準化なし)　複数音高推定 flen=1024
spec_not_normalized_poly = learning_history.from_dir(
    "./experimental_results/da_spec_not_normalized_poly/result_5fold/histories"
)
spec_not_normalized_poly = learning_history.average(*spec_not_normalized_poly)
spec_not_normalized_test_poly = learning_history.from_path(
    "./experimental_results/da_spec_not_normalized_poly/result_5fold/test_res.csv"
)

# 対数スペクトル　複数音高推定
logspec_poly = learning_history.from_dir(
    "./experimental_results/da_logspec_poly/result_5fold/histories"
)
logspec_poly = learning_history.average(*logspec_poly)
logspec_test_poly = learning_history.from_path(
    "./experimental_results/da_logspec_poly/result_5fold/test_res.csv"
)

# 対数スペクトル　単音高推定
logspec_mono = learning_history.from_dir(
    "./experimental_results/da_logspec_mono/result_5fold/histories"
)
logspec_mono = learning_history.average(*logspec_mono)
logspec_test_mono = learning_history.from_path(
    "./experimental_results/da_logspec_mono/result_5fold/test_res.csv"
)

# スペクトル(標準化なし) 複数音高推定 flen=2048
spec_w2048_poly = learning_history.from_dir(
    "./experimental_results/da_spec_w2048_poly/result_5fold/histories"
)
spec_w2048_poly = learning_history.average(*spec_w2048_poly)
spec_w2048_test_poly = learning_history.from_path(
    "./experimental_results/da_spec_w2048_poly/result_5fold/test_res.csv"
)

# スペクトル(標準化なし) 複数音高推定 flen=4096
spec_w4096_poly = learning_history.from_dir(
    "./experimental_results/da_spec_w4096_poly/result_5fold/histories"
)
spec_w4096_poly = learning_history.average(*spec_w4096_poly)
spec_w4096_test_poly = learning_history.from_path(
    "./experimental_results/da_spec_w4096_poly/result_5fold/test_res.csv"
)

# スペクトル(標準化なし) 複数音高推定 flen=512
spec_w512_poly = learning_history.from_dir(
    "./experimental_results/da_spec_w512_poly/result_5fold/histories"
)
spec_w512_poly = learning_history.average(*spec_w512_poly)
spec_w512_test_poly = learning_history.from_path(
    "./experimental_results/da_spec_w512_poly/result_5fold/test_res.csv"
)

metrics = ["precision", "recall", "F1"]

# 複数音高推定での比較
cplot.plot_histories(
    "./comparison_results/da_poly",
    waveform_poly,
    spec_poly,
    logspec_poly,
    metrics=metrics,
    legend=["waveform", "spectrum", "log spectrum"],
)

cplot.box_plot_histories(
    "./comparison_results/da_poly/test_res_compare.png",
    waveform_test_poly,
    spec_test_poly,
    logspec_test_poly,
    metrics=metrics,
    legend=["waveform", "spectrum", "log spectrum"],
)


# 単音高推定での比較
cplot.plot_histories(
    "./comparison_results/da_mono",
    waveform_mono,
    spec_mono,
    logspec_mono,
    metrics=metrics,
    legend=["waveform", "spectrum", "log spectrum"],
)

cplot.box_plot_histories(
    "./comparison_results/da_mono/test_res_compare.png",
    waveform_test_mono,
    spec_test_mono,
    logspec_test_mono,
    metrics=metrics,
    legend=["waveform", "spectrum", "log spectrum"],
)


# 標準化による比較
cplot.plot_histories(
    "./comparison_results/da_normalize_poly",
    spec_poly,
    spec_not_normalized_poly,
    metrics=metrics,
    legend=["normalized", "not normalized"],
)

cplot.box_plot_histories(
    "./comparison_results/da_normalize_poly/test_res_compare.png",
    spec_test_poly,
    spec_not_normalized_test_poly,
    metrics=metrics,
    legend=["normalized", "not normalized"],
)

# フレーム長による比較
cplot.plot_histories(
    "./comparison_results/da_window_len_poly",
    spec_w512_poly,
    spec_not_normalized_poly,
    spec_w2048_poly,
    spec_w4096_poly,
    metrics=metrics,
    legend=["512", "1024", "2048", "4096"],
)

cplot.box_plot_histories(
    "./comparison_results/da_window_len_poly/test_res_compare.png",
    spec_w512_test_poly,
    spec_not_normalized_test_poly,
    spec_w2048_test_poly,
    spec_w4096_test_poly,
    metrics=metrics,
    legend=["512", "1024", "2048", "4096"],
)