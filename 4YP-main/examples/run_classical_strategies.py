import os
from gml.backtest import run_classical_methods

from settings.default import STRADDLE
# INTERVALS = [(2011, y, y + 1) for y in range(2016, 2022)]
# INTERVALS = [(2011, 2021, 2022)]
INTERVALS = [(2011, 2017, 2023)]
# INTERVALS = [(2011, y, y + 1) for y in range(2021, 2024)]


# Original LSTM Strategy
# REFERENCE_EXPERIMENT = "experiment_gml_lstm_cpnone_len63_notime_div_v1"

# Graph ML Strategy
# REFERENCE_EXPERIMENT = "experiment_graph_split80_gml_cpnone_len63_notime_div_v1"
REFERENCE_EXPERIMENT = "quick_test"

if STRADDLE:
    features_file_path = os.path.join(
        "data",
        "straddle_features",
        "features.csv",
    )
else:
    features_file_path = os.path.join(
        "data",
        "equity_features",
        "features.csv",
    )
    

run_classical_methods(features_file_path, INTERVALS, REFERENCE_EXPERIMENT)
