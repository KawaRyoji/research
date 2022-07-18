from experiments.transformer.transformer_experiment import TransformerExperiment
from machine_learning.parameter import hyper_params

params = hyper_params(32, 1000, epoch_size=500, learning_rate=0.0001)

root_dir = "./experimental_results/transformer/mlp"
feature_name = "log_spectrogram"

experiment = TransformerExperiment(root_dir, feature_name, params, flen=1024)
experiment.run()

experiment = TransformerExperiment(
    root_dir, feature_name, params, flen=1024, normalize=True
)
experiment.run()

experiment = TransformerExperiment(root_dir, feature_name, params, flen=2048)
experiment.run()

experiment = TransformerExperiment(
    root_dir, feature_name, params, flen=2048, normalize=True
)
experiment.run()
