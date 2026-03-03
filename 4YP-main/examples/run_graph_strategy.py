import os
import argparse
from settings.hp_grid import HP_MINIBATCH_SIZE_GRAPH
import pandas as pd
from settings.default import ALL_TICKERS
from settings.fixed_params import MODEL_PARAMS_GRAPH
from gml.backtest import run_all_windows
import numpy as np
from functools import reduce

from settings.default import STRADDLE, GRAPH_THRESHOLD, PEARSON

# define the asset class of each ticker here - for this example we have not done this
TEST_MODE = False
PLOT_MODE = True

RUN_SINGLE_WINDOW = True

ASSET_CLASS_MAPPING = dict(zip(ALL_TICKERS, ["V"] * len(ALL_TICKERS)))
TRAIN_VALID_RATIO = 0.8
TIME_FEATURES = False
FORCE_OUTPUT_SHARPE_LENGTH = None
EVALUATE_DIVERSIFIED_VAL_SHARPE = False # was True? -> turns val positive? not sure whats going on here
NAME = "exp_lstm_gcn"


def main(
    experiment: str,
    train_start: int,
    test_start: int,
    test_end: int,
    test_window_size: int,
    num_repeats: int,
):
    # Set parameters based on experiment type.
    # if experiment == "LSTM":
    #     architecture = "LSTM"
    #     time_steps = 63
    #     changepoint_lbws = None
    if experiment == "LSTM-GCN":
        architecture = "LSTM-GCN"
        time_steps = 20 # OG: 63
        changepoint_lbws = None
    elif experiment == "LSTM-GCN_BENCHMARK":
        architecture = "LSTM-GCN_BENCHMARK"
        time_steps = 20
        changepoint_lbws = None
    else:
        raise BaseException("Invalid experiment.")

    versions = range(1, 1 + num_repeats) if not TEST_MODE else [1]

    experiment_prefix = (
        NAME
        + ("_TEST" if TEST_MODE else "")
        + ("" if TRAIN_VALID_RATIO == 0.90 else f"_split{int(TRAIN_VALID_RATIO * 100)}")
    )

    cp_string = (
        "none"
        if not changepoint_lbws
        else reduce(lambda x, y: str(x) + str(y), changepoint_lbws)
    )
    time_string = "time" if TIME_FEATURES else "notime"
    _project_name = f"{experiment_prefix}_{architecture.lower()}_cp{cp_string}_len{time_steps}_{time_string}_{'div' if EVALUATE_DIVERSIFIED_VAL_SHARPE else 'val'}"
    if FORCE_OUTPUT_SHARPE_LENGTH:
        _project_name += f"_outlen{FORCE_OUTPUT_SHARPE_LENGTH}"
    _project_name += "_v"

    for v in versions:
        PROJECT_NAME = _project_name + str(v)

        # intervals = [
        #     (train_start, y, y + test_window_size)
        #     for y in range(test_start, test_end - 1)
        # ]
        
        if RUN_SINGLE_WINDOW:
            # Only use the first interval (i.e. for y = test_start)
            intervals = [(train_start, test_start, test_start + test_window_size)]
        else:
            # Use all intervals
            intervals = [
                (train_start, y, y + test_window_size)
                for y in range(test_start, test_end - 1)
            ]

        params = MODEL_PARAMS_GRAPH.copy()
        params["num_tickers"] = len(ALL_TICKERS)
        params["total_time_steps"] = time_steps
        params["architecture"] = architecture
        params["evaluate_diversified_val_sharpe"] = EVALUATE_DIVERSIFIED_VAL_SHARPE
        params["train_valid_ratio"] = TRAIN_VALID_RATIO
        params["time_features"] = TIME_FEATURES
        params["force_output_sharpe_length"] = FORCE_OUTPUT_SHARPE_LENGTH

        if TEST_MODE:
            params["num_epochs"] = 1
            params["random_search_iterations"] = 2

        if STRADDLE:
            features_file_path = os.path.join(
                "data", "straddle_features", "features.csv"
            )
        else:
            features_file_path = os.path.join("data", "equity_features", "features.csv")

        # For LSTM experiments, a different minibatch grid may be used.
        # Here, we choose HP_MINIBATCH_SIZE unless time_steps equals 252 (for LSTM-specific cases).
        minibatch_choice = [32, 64, 128] if time_steps == 252 else HP_MINIBATCH_SIZE_GRAPH

        # if PEARSON:
        #     graph_file_path = os.path.join("data", "graph_structure", "pearson", f"{GRAPH_THRESHOLD}.csv")

        run_all_windows(
            PROJECT_NAME,
            features_file_path,
            intervals,
            params,
            ASSET_CLASS_MAPPING,
            minibatch_choice,
            test_window_size,
        )

        # if PLOT_MODE:
        #    ... (plotting code if needed) ...


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""
        parser = argparse.ArgumentParser(description="Run DMN experiment")
        parser.add_argument(
            "experiment",
            metavar="c",
            type=str,
            nargs="?",
            default="LSTM-GCN",
            choices=[
                "LSTM",
                "LSTM-GCN",
                "LSTM-GCN_BENCHMARK",
                "LSTM-GAT",
                # other choices if needed...
            ],
            help="Experiment type: LSTM or GML (Graph Neural Network)",
        )
        parser.add_argument(
            "train_start",
            metavar="s",
            type=int,
            nargs="?",
            default=2011,
            help="Training start year",
        )
        parser.add_argument(
            "test_start",
            metavar="t",
            type=int,
            nargs="?",
            default=2017, # Try -> 2017
            help="Training end year and test start year.",
        )
        parser.add_argument(
            "test_end",
            metavar="e",
            type=int,
            nargs="?",
            default=2023, 
            help="Testing end year.",
        )
        parser.add_argument(
            "test_window_size",
            metavar="w",
            type=int,
            nargs="?",
            default=6, # Try -> 2
            help="Test window length in years.",
        )
        parser.add_argument(
            "num_repeats",
            metavar="r",
            type=int,
            nargs="?",
            default=1,
            help="Number of experiment repeats.",
        )

        args = parser.parse_known_args()[0]
        return (
            args.experiment,
            args.train_start,
            args.test_start,
            args.test_end,
            args.test_window_size,
            args.num_repeats,
        )

    main(*get_args())
