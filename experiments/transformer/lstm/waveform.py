from experiments.transformer.transformer_experiment import TransformerExperiment
from machine_learning.parameter import hyper_params
from tensorflow.keras.layers import LSTM, Permute
from tensorflow.keras.models import Sequential

params = hyper_params(32, 1000, epoch_size=500, learning_rate=0.0001)

root_dir = "./experimental_results/transformer/lstm"
feature_name = "waveform"
decoder = Sequential(
    layers=[Permute((2, 1)), LSTM(32, return_sequences=True), Permute((2, 1))],
    name="decoder",
)

experiment = TransformerExperiment(root_dir, feature_name, params, decoder=decoder)
experiment.run()

decoder.reset_states()

experiment = TransformerExperiment(
    root_dir, feature_name, params, normalize=True, decoder=decoder
)
experiment.run()
