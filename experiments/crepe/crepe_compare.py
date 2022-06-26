import machine_learning.comparison_plot as cplt
from machine_learning.learning_history import learning_history


# 時間波形　単音高推定
waveform_mono = learning_history.from_dir(
    "./experimental_results/crepe_waveform_mono/result_5fold/histories"
)
waveform_mono = learning_history.average(*waveform_mono)
waveform_test_mono = learning_history.from_path(
    "./experimental_results/crepe_waveform_mono/result_5fold/test_res.csv"
)

# 時間波形　複数音高推定
waveform_poly = learning_history.from_dir(
    "./experimental_results/crepe_waveform_poly/result_5fold/histories"
)
waveform_poly = learning_history.average(*waveform_poly)
waveform_test_poly = learning_history.from_path(
    "./experimental_results/crepe_waveform_poly/result_5fold/test_res.csv"
)

# 時間波形　複数音高推定　上限なし
waveform_nolimit = learning_history.from_path(
    "./experimental_results/crepe_waveform_nolimit_poly/holdout_result/history.csv"
)
waveform_test_nolimit = learning_history.from_path(
    "./experimental_results/crepe_waveform_nolimit_poly/holdout_result/test_res.csv"
)

# スペクトル　単音高推定
spec_mono = learning_history.from_dir(
    "./experimental_results/crepe_spec_mono/result_5fold/histories"
)
spec_mono = learning_history.average(*spec_mono)
spec_test_mono = learning_history.from_path(
    "./experimental_results/crepe_spec_mono/result_5fold/test_res.csv"
)

# スペクトル　複数音高推定
spec_poly = learning_history.from_dir(
    "./experimental_results/crepe_spec_poly/result_5fold/histories"
)
spec_poly = learning_history.average(*spec_poly)
spec_test_poly = learning_history.from_path(
    "./experimental_results/crepe_spec_poly/result_5fold/test_res.csv"
)

# スペクトル　複数音高推定　上限なし
spec_nolimit = learning_history.from_path(
    "./experimental_results/crepe_spec_nolimit_poly/holdout_result/history.csv"
)
spec_test_nolimit = learning_history.from_path(
    "./experimental_results/crepe_spec_nolimit_poly/holdout_result/test_res.csv"
)

# 対数スペクトル　単音高推定
# logspec_mono = learning_history.from_dir(
#     "./experimental_results/crepe_logspec_mono/result_5fold/histories"
# )
# logspec_mono = learning_history.average(*logspec_mono)
# logspec_test_mono = learning_history.from_path(
#     "./experimental_results/crepe_logspec_mono/result_5fold/test_res.csv"
# )

# 対数スペクトル　複数音高推定
logspec_poly = learning_history.from_dir(
    "./experimental_results/crepe_logspec_poly/result_5fold/histories"
)
logspec_poly = learning_history.average(*logspec_poly)
logspec_test_poly = learning_history.from_path(
    "./experimental_results/crepe_logspec_poly/result_5fold/test_res.csv"
)

# 対数スペクトル　複数音高推定　上限なし
logspec_nolimit = learning_history.from_path(
    "./experimental_results/crepe_logspec_nolimit_poly/holdout_result/history.csv"
)
logspec_test_nolimit = learning_history.from_path(
    "./experimental_results/crepe_logspec_nolimit_poly/holdout_result/test_res.csv"
)


