import os
from typing import Tuple, List, Dict
import tensorflow as tf
import pandas as pd
import datetime as dt
import numpy as np
import shutil
import gc
import copy

import json
from settings.default import ALL_TICKERS, GRAPH_THRESHOLD, PEARSON

from gml.data_prep import MACDStrategy

from gml.model_inputs import ModelFeatures
from gml.graph_model_inputs import GraphModelFeatures
from gml.deep_neural_network import LstmDeepMomentumNetworkModel
# from gml.graph_model import GraphDeepMomentumModel, LstmGATDeepMomentumNetworkModel, GCLstmDeepMomentumNetworkModel
from gml.graph_model_2 import GraphLSTMDeepMomentumNetwork
from gml.classical_strategies import (
    VOL_TARGET,
    calc_performance_metrics,
    calc_performance_metrics_subset,
    calc_sharpe_by_year,
    calc_vol_scaled_returns,
    calc_net_returns,
    annual_volatility,
)

from settings.default import BACKTEST_AVERAGE_BASIS_POINTS

from settings.hp_grid import HP_MINIBATCH_SIZE, HP_MINIBATCH_SIZE_GRAPH, HP_ALPHA, HP_BETA

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def _get_directory_name(
    experiment_name: str, train_interval: Tuple[int, int, int] = None
) -> str:
    """The directory name for saving results

    Args:
        experiment_name (str): name of experiment
        train_interval (Tuple[int, int, int], optional): (start yr, end train yr / start test yr, end test year) Defaults to None.

    Returns:
        str: folder name
    """
    if train_interval:
        return os.path.join(
            "results", experiment_name, f"{train_interval[1]}-{train_interval[2]}"
        )
    else:
        return os.path.join(
            "results",
            experiment_name,
        )


def _basis_point_suffix(basis_points: float = None) -> str:
    """Basis points suffix

    Args:
        basis_points (float, optional): bps valud. Defaults to None.

    Returns:
        str: suffix name
    """
    if not basis_points:
        return ""
    return "_" + str(basis_points).replace(".", "_") + "_bps"


def _interval_suffix(
    train_interval: Tuple[int, int, int], basis_points: float = None
) -> str:
    """Interval points suffix

    Args:
        train_interval (Tuple[int, int, int], optional): (start yr, end train yr / start test yr, end test year) Defaults to None.
        basis_points (float, optional): bps valud. Defaults to None.

    Returns:
        str: suffix name
    """
    return f"_{train_interval[1]}_{train_interval[2]}" + _basis_point_suffix(
        basis_points
    )


def _results_from_all_windows(
    experiment_name: str, train_intervals: List[Tuple[int, int, int]]
):
    """Save a json with results from all windows

    Args:
        experiment_name (str): experiment name
        train_intervals (List[Tuple[int, int, int]]): list of training intervals
    """
    return pd.concat(
        [
            pd.read_json(
                os.path.join(
                    _get_directory_name(experiment_name, interval), "results.json"
                ),
                # typ="series",
            )
            for interval in train_intervals
        ]
    )


def _get_asset_classes(asset_class_dictionary: Dict[str, str]):
    return np.unique(list(asset_class_dictionary.values())).tolist()


def _captured_returns_from_all_windows(
    experiment_name: str,
    train_intervals: List[Tuple[int, int, int]],
    volatility_rescaling: bool = True,
    only_standard_windows: bool = True,
    volatilites_known: List[float] = None,
    filter_identifiers: List[str] = None,
    captured_returns_col: str = "captured_returns",
    standard_window_size: int = 1,
) -> pd.Series:
    """get sereis of captured returns from all intervals

    Args:
        experiment_name (str): name of experiment
        train_intervals (List[Tuple[int, int, int]]): list of training intervals
        volatility_rescaling (bool, optional): rescale to target annualised volatility. Defaults to True.
        only_standard_windows (bool, optional): only include full windows. Defaults to True.
        volatilites_known (List[float], optional): list of annualised volatities, if known. Defaults to None.
        filter_identifiers (List[str], optional): only run for specified tickers. Defaults to None.
        captured_returns_col (str, optional): column name of captured returns. Defaults to "captured_returns".
        standard_window_size (int, optional): number of years in standard window. Defaults to 1.

    Returns:
        pd.Series: series of captured returns
    """
    srs_list = []
    volatilites = volatilites_known if volatilites_known else []
    for interval in train_intervals:
        print(interval[2])
        print(interval[1])
        print(standard_window_size)
        # if only_standard_windows and (
        #     interval[2] - interval[1] == standard_window_size
        # ):
        if only_standard_windows:
            df = pd.read_csv(
                os.path.join(
                    _get_directory_name(experiment_name, interval),
                    "captured_returns_sw.csv",
                ),
            )
            print(df)
            if filter_identifiers:
                filter = pd.DataFrame({"identifier": filter_identifiers})
                df = df.merge(filter, on="identifier")
            num_identifiers = len(df["identifier"].unique())
            srs = df.groupby("time")[captured_returns_col].sum() / num_identifiers
            srs_list.append(srs)
            if volatility_rescaling and not volatilites_known:
                volatilites.append(annual_volatility(srs))
                
                
        else:
            print("HELP")
    if volatility_rescaling:
        return pd.concat(srs_list) * VOL_TARGET / np.mean(volatilites)
    else:
        return pd.concat(srs_list)


def save_results(
    results_sw: pd.DataFrame,
    output_directory: str,
    train_interval: Tuple[int, int, int],
    num_identifiers: int,
    asset_class_dictionary: Dict[str, str],
    extra_metrics: dict = {},
):
    """save results json

    Args:
        results_sw (pd.DataFrame): results dataframe
        output_directory (str): output directory
        train_interval (Tuple[int, int, int]): training interval
        num_identifiers (int): number of tickers
        asset_class_dictionary (Dict[str, str]): mapping of ticker to asset class
        extra_metrics (dict, optional): additional metrics to save. Defaults to {}.
    """
    asset_classes = ["ALL"]
    results_asset_class = [results_sw]
    if asset_class_dictionary:
        results_sw["asset_class"] = results_sw["identifier"].map(
            lambda i: asset_class_dictionary[i]
        )
        classes = _get_asset_classes(asset_class_dictionary)
        for ac in classes:
            results_asset_class += [results_sw[results_sw["asset_class"] == ac]]
        asset_classes += classes

    metrics = {}
    for ac, results_ac in zip(asset_classes, results_asset_class):
        suffix = _interval_suffix(train_interval)
        if ac == "ALL" and extra_metrics:
            ac_metrics = extra_metrics.copy()
        else:
            ac_metrics = {}
        for basis_points in BACKTEST_AVERAGE_BASIS_POINTS:
            suffix = _interval_suffix(train_interval, basis_points)
            if basis_points:
                results_ac_bps = results_ac.drop(columns="captured_returns").rename(
                    columns={
                        "captured_returns"
                        + _basis_point_suffix(basis_points): "captured_returns"
                    }
                )
            else:
                results_ac_bps = results_ac

            ac_metrics = {
                **ac_metrics,
                **calc_performance_metrics(
                    results_ac_bps.set_index("time"), suffix, num_identifiers
                ),
                **calc_sharpe_by_year(
                    results_ac_bps.set_index("time"), _basis_point_suffix(basis_points)
                ),
            }
        metrics = {**metrics, ac: ac_metrics}

    with open(os.path.join(output_directory, "results.json"), "w") as file:
        file.write(json.dumps(metrics, indent=4))


def aggregate_and_save_all_windows(
    experiment_name: str,
    train_intervals: List[Tuple[int, int, int]],
    asset_class_dictionary: Dict[str, str],
    standard_window_size: int,
):
    """Save a results summary, aggregating all windows

    Args:
        experiment_name (str): experiment name
        train_intervals (List[Tuple[int, int, int]]): list of train/test intervals
        asset_class_dictionary (Dict[str, str]): map tickers to asset class
        standard_window_size (int): number of years in standard window
    """
    directory = _get_directory_name(experiment_name)
    all_results = _results_from_all_windows(experiment_name, train_intervals)

    _metrics = [
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "downside_risk",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "perc_pos_return",
        "profit_loss_ratio",
    ]
    _rescaled_metrics = [
        "annual_return_rescaled",
        "annual_volatility_rescaled",
        "downside_risk_rescaled",
        "max_drawdown_rescaled",
    ]

    metrics = []
    rescaled_metrics = []
    for bp in BACKTEST_AVERAGE_BASIS_POINTS:
        suffix = _basis_point_suffix(bp)
        metrics += list(map(lambda m: m + suffix, _metrics))
        rescaled_metrics += list(map(lambda m: m + suffix, _rescaled_metrics))

    if asset_class_dictionary:
        asset_classes = ["ALL"] + _get_asset_classes(asset_class_dictionary)
    else:
        asset_classes = ["ALL"]

    average_metrics = {}
    list_metrics = {}

    asset_class_tickers = (
        pd.DataFrame.from_dict(asset_class_dictionary, orient="index")
        .reset_index()
        .set_index(0)
    )
    
   

    for asset_class in asset_classes:
        average_results = dict(
            zip(
                metrics + rescaled_metrics,
                [[] for _ in range(len(metrics + rescaled_metrics))],
            )
        )
        asset_results = all_results[asset_class]

        for bp in BACKTEST_AVERAGE_BASIS_POINTS:
            suffix = _basis_point_suffix(bp)
            average_results[f"sharpe_ratio_years{suffix}"] = []
        # average_results["sharpe_ratio_years_std"] = 0.0

        for interval in train_intervals:
            # only want full windows here
            if interval[2] - interval[1] == standard_window_size:
                for m in _metrics:
                    for bp in BACKTEST_AVERAGE_BASIS_POINTS:
                        suffix = _interval_suffix(interval, bp)
                        average_results[m + _basis_point_suffix(bp)].append(
                            asset_results[m + suffix]
                        )

            for bp in BACKTEST_AVERAGE_BASIS_POINTS:
                suffix = _basis_point_suffix(bp)
                for year in range(interval[1], interval[2]):
                    average_results["sharpe_ratio_years" + suffix].append(
                        asset_results[f"sharpe_ratio_{int(year)}{suffix}"]
                    )
        for bp in BACKTEST_AVERAGE_BASIS_POINTS:
            suffix = _basis_point_suffix(bp)
            all_captured_returns = _captured_returns_from_all_windows(
                experiment_name,
                train_intervals,
                volatility_rescaling=True,
                only_standard_windows=True,
                volatilites_known=average_results["annual_volatility" + suffix],
                filter_identifiers=(
                    None
                    if asset_class == "ALL"
                    else asset_class_tickers.loc[
                        asset_class, asset_class_tickers.columns[0]
                    ].tolist()
                ),
                captured_returns_col=f"captured_returns{suffix}",
                standard_window_size=6, # TODO; HARDCODED to work with 2017,2023!!!!
            )
            yrs = pd.to_datetime(all_captured_returns.index).year
            for interval in train_intervals:
                if interval[2] - interval[1] == standard_window_size:
                    srs = all_captured_returns[
                        (yrs >= interval[1]) & (yrs < interval[2])
                    ]
                    rescaled_dict = calc_performance_metrics_subset(
                        srs, f"_rescaled{suffix}"
                    )
                    for m in _rescaled_metrics:
                        average_results[m + suffix].append(rescaled_dict[m + suffix])

        window_history = copy.deepcopy(average_results)
        for key in average_results:
            average_results[key] = np.mean(average_results[key])

        for bp in BACKTEST_AVERAGE_BASIS_POINTS:
            suffix = _basis_point_suffix(bp)
            average_results[f"sharpe_ratio_years_std{suffix}"] = np.std(
                window_history[f"sharpe_ratio_years{suffix}"]
            )

        average_metrics = {**average_metrics, asset_class: average_results}
        list_metrics = {**list_metrics, asset_class: window_history}

    with open(os.path.join(directory, "average_results.json"), "w") as file:
        file.write(json.dumps(average_metrics, indent=4))
    with open(os.path.join(directory, "list_results.json"), "w") as file:
        file.write(json.dumps(list_metrics, indent=4))


def run_single_window(
    experiment_name: str,
    features_file_path: str,
    train_interval: Tuple[int, int, int],
    params: dict,
    # changepoint_lbws: List[int],
    skip_if_completed: bool = True,
    asset_class_dictionary: Dict[str, str] = None,
    hp_minibatch_size: List[int] = HP_MINIBATCH_SIZE, # TODO i changed this?
):
    """Backtest for a single test window

    Args:
        experiment_name (str): experiment name
        features_file_path (str): name of file, containing features
        train_interval (Tuple[int, int, int], optional): (start yr, end train yr / start test yr, end test year)
        params (dict): dmn experiment parameters
        changepoint_lbws (List[int]): CPD LBWs to be used
        skip_if_completed (bool, optional): skip, if previously completed. Defaults to True.
        asset_class_dictionary (Dict[str, str], optional): map tickers to asset class. Defaults to None.
        hp_minibatch_size (List[int], optional): minibatch size hyperparameter grid. Defaults to HP_MINIBATCH_SIZE.

    Raises:
        Exception: [description]
    """
    directory = _get_directory_name(experiment_name, train_interval)

    if skip_if_completed and os.path.exists(os.path.join(directory, "results.json")):
        print(
            f"Skipping {train_interval[1]}-{train_interval[2]} because already completed."
        )
        return

    raw_data = pd.read_csv(features_file_path, index_col=0, parse_dates=True)
    raw_data["date"] = raw_data["date"].astype("datetime64[ns]")


    # TODO more/less than the one year test buffer
    
    if params["architecture"] == "LSTM":
        model_features = ModelFeatures(
            raw_data,
            params["total_time_steps"],
            start_boundary=train_interval[0],
            test_boundary=train_interval[1],
            test_end=train_interval[2],
            # changepoint_lbws=changepoint_lbws,
            split_tickers_individually=params["split_tickers_individually"],
            train_valid_ratio=params["train_valid_ratio"],
            add_ticker_as_static=(params["architecture"] == "TFT"),
            time_features=params["time_features"],
            lags=params["force_output_sharpe_length"],
            asset_class_dictionary=asset_class_dictionary,
        )
    
    elif params["architecture"] == "LSTM-GCN":
        model_features = GraphModelFeatures(
            raw_data,
            params["total_time_steps"],
            start_boundary=train_interval[0],
            test_boundary=train_interval[1],
            test_end=train_interval[2],
            # changepoint_lbws=changepoint_lbws,
            split_tickers_individually=params["split_tickers_individually"],
            train_valid_ratio=params["train_valid_ratio"],
            add_ticker_as_static=(params["architecture"] == "TFT"),
            time_features=params["time_features"],
            lags=params["force_output_sharpe_length"],
            asset_class_dictionary=asset_class_dictionary,
        )
        
    elif params["architecture"] == "LSTM-GCN_BENCHMARK":
        model_features = GraphModelFeatures(
            raw_data,
            params["total_time_steps"],
            start_boundary=train_interval[0],
            test_boundary=train_interval[1],
            test_end=train_interval[2],
            # changepoint_lbws=changepoint_lbws,
            split_tickers_individually=params["split_tickers_individually"],
            train_valid_ratio=params["train_valid_ratio"],
            add_ticker_as_static=(params["architecture"] == "TFT"),
            time_features=params["time_features"],
            lags=params["force_output_sharpe_length"],
            asset_class_dictionary=asset_class_dictionary,
        )

    hp_directory = os.path.join(directory, "hp")

    if params["architecture"] == "LSTM":
        dmn = LstmDeepMomentumNetworkModel(
            experiment_name,
            hp_directory,
            hp_minibatch_size,
            **params,
            **model_features.input_params,
        )
        
    elif params["architecture"] == "LSTM-GCN":
        # if PEARSON:
        #     graph_dir = os.path.join("data", "graph_structure", "pearson", f"{GRAPH_THRESHOLD}.csv")
        # else:
        #     graph_dir = os.path.join("data", "graph_structure", "cvx_opt", f"{HP_ALPHA}_{HP_BETA}_cvx".csv")
        dmn = GraphLSTMDeepMomentumNetwork(
            experiment_name,
            hp_directory,
            hp_minibatch_size=HP_MINIBATCH_SIZE_GRAPH,
            # graph_directory=graph_dir,
            **params,
            **model_features.input_params,
        )
        
    elif params["architecture"] == "LSTM-GCN_BENCHMARK":
        # graph_dir = os.path.join("data", "graph_structure", "identity.csv")
        dmn = GraphLSTMDeepMomentumNetwork(
            experiment_name,
            hp_directory,
            hp_minibatch_size=HP_MINIBATCH_SIZE_GRAPH,
            # graph_directory=graph_dir,
            **params,
            **model_features.input_params,
        )

    else:
        dmn = None
        raise Exception(f"{params['architecture']} is not a valid architecture.")

    
    best_hp, best_model, training_history = dmn.hyperparameter_search(
        model_features.train, model_features.valid
    )
    
    ########################################
    history_file_path = os.path.join(directory, "training_history.json")
    with open(history_file_path, "w") as file:
        json.dump(training_history.history, file)
        
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(training_history.history['loss'], label='Training Loss')
    plt.plot(training_history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plot_file_path = os.path.join(directory, "loss_plot.png")
    plt.savefig(plot_file_path)
    plt.close()
    print(f"Loss plot saved to {plot_file_path}")
    #########################################
        
    print(best_hp)
    print(best_model)
    val_loss = dmn.evaluate(model_features.valid, best_model)

    print(f"Best validation loss = {val_loss}")
    print(f"Best params:")
    for k in best_hp:
        print(f"{k} = {best_hp[k]}")

    with open(os.path.join(directory, "best_hyperparameters.json"), "w") as file:
        file.write(json.dumps(best_hp))

    # if predict_on_test_set:
    print("Predicting on test set...")

    results_sw, performance_sw = dmn.get_positions(
        model_features.test_sliding,
        best_model,
        sliding_window=True,
        years_geq=train_interval[1],
        years_lt=train_interval[2],
    )
    print(f"performance (sliding window) = {performance_sw}")

    results_sw = results_sw.merge(
        raw_data.reset_index()[["ticker", "date", "daily_vol"]].rename(
            columns={"ticker": "identifier", "date": "time"}
        ),
        on=["identifier", "time"],
    )
    results_sw = calc_net_returns(
        results_sw, BACKTEST_AVERAGE_BASIS_POINTS[1:], model_features.tickers
    )
    results_sw.to_csv(os.path.join(directory, "captured_returns_sw.csv"))

    # keep fixed window just in case
    results_fw, performance_fw = dmn.get_positions(
        model_features.test_fixed,
        best_model,
        sliding_window=False,
        years_geq=train_interval[1],
        years_lt=train_interval[2],
    )
    print(f"performance (fixed window) = {performance_fw}")
    results_fw = results_fw.merge(
        raw_data.reset_index()[["ticker", "date", "daily_vol"]].rename(
            columns={"ticker": "identifier", "date": "time"}
        ),
        on=["identifier", "time"],
    )
    
    # TODO add vol scaled returns?
    # results_fw = calc_vol_scaled_returns(
    #     results_fw
    
    # def calc_vol_scaled_returns(daily_returns, daily_vol=pd.Series(None)):
    # """calculates volatility scaled returns for annualised VOL_TARGET of 15%
    # with input of pandas series daily_returns"""
    # if not len(daily_vol):
    #     daily_vol = calc_daily_vol(daily_returns)
    # annualised_vol = daily_vol * np.sqrt(252)  # annualised
    # return daily_returns * VOL_TARGET / annualised_vol.shift(1)
    
    
    results_fw = calc_net_returns(
        results_fw, BACKTEST_AVERAGE_BASIS_POINTS[1:], model_features.tickers
    )
    results_fw.to_csv(os.path.join(directory, "captured_returns_fw.csv"))

    with open(os.path.join(directory, "fixed_params.json"), "w") as file:
        file.write(
            json.dumps(
                dict(
                    **params,
                    **model_features.input_params,
                    **{
                        "features_file_path": features_file_path,
                    },
                ),
                indent=4,
            )
        )

    # save model and get rid of the hp dir
    best_directory = os.path.join(directory, "best")
    best_model.save_weights(os.path.join(best_directory, "checkpoints", "checkpoint.weights.h5"))
    with open(os.path.join(best_directory, "hyperparameters.json"), "w") as file:
        file.write(json.dumps(best_hp, indent=4))
    shutil.rmtree(hp_directory)

    save_results(
        results_sw,
        directory,
        train_interval,
        model_features.num_tickers,
        asset_class_dictionary,
        {
            "performance_sw": performance_sw,
            "performance_fw": performance_fw,
            "val_loss": val_loss,
        },
    )

    # get rid of everything and reset - TODO maybe not needed...
    del best_model
    gc.collect()
    tf.keras.backend.clear_session()
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def run_all_windows(
    experiment_name: str,
    features_file_path: str,
    train_intervals: List[Tuple[int, int, int]],
    params: dict,
    # changepoint_lbws: List[int],
    asset_class_dictionary=Dict[str, str],
    hp_minibatch_size=HP_MINIBATCH_SIZE,
    standard_window_size=1,
):
    """Run experiment for multiple test intervals and aggregate results

    Args:
        experiment_name (str): experiment name
        features_file_path (str): name of file, containing features
        train_intervals (List[Tuple[int, int, int]]): klist of all training intervals
        params (dict): dmn experiment parameters
        changepoint_lbws (List[int]): CPD LBWs to be used
        asset_class_dictionary ([type], optional): map tickers to asset class. Defaults to None. Defaults to Dict[str, str].
        hp_minibatch_size ([type], optional): minibatch size hyperparameter grid. Defaults to HP_MINIBATCH_SIZE.
        standard_window_size (int, optional): standard number of years in test window. Defaults to 1.
    """
    if params["architecture"]=="LSTM-GCN":
        hp_minibatch_size=HP_MINIBATCH_SIZE_GRAPH
    # run the expanding window
    for interval in train_intervals:
        run_single_window(
            experiment_name,
            features_file_path,
            interval,
            params,
            # changepoint_lbws,
            asset_class_dictionary=asset_class_dictionary,
            hp_minibatch_size=hp_minibatch_size,
        )

    aggregate_and_save_all_windows(
        experiment_name, 
        train_intervals, 
        asset_class_dictionary,
        standard_window_size
    )


def intermediate_momentum_position(w: float, returns_data: pd.DataFrame) -> pd.Series:
    """Position size for intermediate strategy.
    See https://arxiv.org/pdf/2105.13727.pdf

    Args:
        w (float): intermediate wweighting
        returns_data (pd.DataFrame): [description]

    Returns:
        pd.Series: series of position sizes
    """
    return w * np.sign(returns_data["norm_monthly_return"]) + (1 - w) * np.sign(
        returns_data["norm_annual_return"]
    )


# def run_classical_methods(
#     features_file_path,
#     train_intervals,
#     reference_experiment,
#     long_only_experiment_name="long_only",
#     tsmom_experiment_name="tsmom",
#     # macd_experiment_name="macd",
    
#     # tsm TODO add mean reversion etc.
# ):
#     """Run classical TSMOM method and Long Only as defined in https://arxiv.org/pdf/2105.13727.pdf.

#     Args:
#         features_file_path: File path containing the features.
#         train_intervals: List of train/test intervals.
#         reference_experiment: Name of the reference experiment.
#         long_only_experiment_name (str, optional): Name of the long only experiment. Defaults to "long_only".
#         tsmom_experiment_name (str, optional): Name of the TSMOM experiment. Defaults to "tsmom".
#     """
    
#     ###################################
#     _metrics = [
#         "annual_return",
#         "annual_volatility",
#         "sharpe_ratio",
#         "downside_risk",
#         "sortino_ratio",
#         "max_drawdown",
#         "calmar_ratio",
#         "perc_pos_return",
#         "profit_loss_ratio",
#     ]
#     _rescaled_metrics = [
#         "annual_return_rescaled",
#         "annual_volatility_rescaled",
#         "downside_risk_rescaled",
#         "max_drawdown_rescaled",
#     ]
#     ####################################
    
    
    
    
    
#     # Create main directories recursively
#     directory = _get_directory_name(long_only_experiment_name)
#     os.makedirs(directory, exist_ok=True)

#     directory = _get_directory_name(tsmom_experiment_name)
#     os.makedirs(directory, exist_ok=True)
    
#     # directory = _get_directory_name(macd_experiment_name)
#     # os.makedirs(directory, exist_ok=True)

#     for train_interval in train_intervals:
#         # Load raw data and merge with reference data once per interval
#         raw_data = pd.read_csv(features_file_path, parse_dates=True)
        
#         # Build the reference file path and check if it exists
#         ref_file_path = os.path.join(
#             "results",
#             reference_experiment,
#             f"{train_interval[1]}-{train_interval[2]}",
#             "captured_returns_sw.csv"
#         )
#         print(ref_file_path)
#         if not os.path.exists(ref_file_path):
#             print(f"Reference file {ref_file_path} not found. Skipping interval {train_interval}.")
#             continue

#         reference = pd.read_csv(ref_file_path, parse_dates=True)
#         base_returns_data = raw_data.merge(
#             reference[["time", "identifier", "returns"]],
#             left_on=["date", "ticker"],
#             right_on=["time", "identifier"],
#         )
        
#         # Run three tsmom experiments with momentum parameters 0, 0.5, and 1.
#         for momentum_val in [0, 0.5, 1]:
#             # Create a modified experiment name to differentiate between momentum values
#             tsmom_experiment_variant = f"{tsmom_experiment_name}_{momentum_val}"
#             directory = _get_directory_name(tsmom_experiment_variant, train_interval)
#             if not os.path.exists(directory):
#                 os.makedirs(directory, exist_ok=True)

#             # Work on a copy of the merged data
#             returns_data = base_returns_data.copy()
#             returns_data["position"] = intermediate_momentum_position(momentum_val, returns_data)
#             returns_data["captured_returns"] = returns_data["position"] * returns_data["returns"]
#             returns_data = returns_data.reset_index()[["identifier", "time", "returns", "position", "captured_returns"]]
#             returns_data.to_csv(os.path.join(directory, "captured_returns_sw.csv"), index=False)

#         directory = _get_directory_name(long_only_experiment_name, train_interval)
#         os.makedirs(directory, exist_ok=True)
#         returns_data = base_returns_data.copy()
#         returns_data["position"] = 1.0
#         returns_data["captured_returns"] = returns_data["position"] * returns_data["returns"]
#         returns_data = returns_data.reset_index()[["identifier", "time", "returns", "position", "captured_returns"]]
#         returns_data.to_csv(os.path.join(directory, "captured_returns_sw.csv"), index=False)

# Added metric calculation & MACD signal




def collect_strategy_metrics(experiment_directories):
    """
    Collects metrics from all experiment directories, averages them per strategy variant,
    and returns a DataFrame where each row corresponds to one strategy.
    
    Args:
        experiment_directories (dict): Dictionary mapping strategy variants to a list of directories.
        
    Returns:
        pd.DataFrame: DataFrame with one row per strategy variant and columns as metric names.
    """
    rows = []
    for variant, dirs in experiment_directories.items():
        metrics_list = []
        for d in dirs:
            m_file = os.path.join(d, "metrics.csv")
            if os.path.exists(m_file):
                df = pd.read_csv(m_file)
                metrics_list.append(df)
        if metrics_list:
            # Concatenate the metrics for all intervals and compute the mean per metric.
            combined = pd.concat(metrics_list, ignore_index=True)
            mean_metrics = combined.mean().to_dict()
            mean_metrics["strategy"] = variant
            rows.append(mean_metrics)
    return pd.DataFrame(rows)





def aggregate_strategy_metrics(experiment_name, train_intervals, experiment_directories, output_directory,
                               asset_class_dictionary=None, standard_window_size=6):
    """
    Aggregates the metrics computed in individual experiment runs and writes out
    average and list results as JSON files.
    
    Args:
        experiment_name: Name of the overall experiment.
        train_intervals: List of train/test intervals.
        experiment_directories: Dictionary mapping experiment variants to a list of directories.
        output_directory: Directory where the JSON files will be saved.
        asset_class_dictionary (dict, optional): Dictionary defining asset classes.
        standard_window_size (int, optional): The window size to consider for aggregation.
    """
    # Define base metric names (without suffix)
    _metrics = [
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "downside_risk",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "perc_pos_return",
        "profit_loss_ratio",
    ]
    _rescaled_metrics = [
        "annual_return_rescaled",
        "annual_volatility_rescaled",
        "downside_risk_rescaled",
        "max_drawdown_rescaled",
    ]
    # Example basis points used for averaging; adjust as needed.
    BACKTEST_AVERAGE_BASIS_POINTS = [0]

    def _basis_point_suffix(bp):
        return f"_{bp}" if bp != 0 else ""

    def _interval_suffix(interval, bp):
        return f"_{interval[1]}-{interval[2]}{_basis_point_suffix(bp)}"

    # Build lists of metric names (with basis point suffixes)
    metrics = []
    rescaled_metrics = []
    for bp in BACKTEST_AVERAGE_BASIS_POINTS:
        suffix = _basis_point_suffix(bp)
        metrics += [m + suffix for m in _metrics]
        rescaled_metrics += [m + suffix for m in _rescaled_metrics]
    
    # For simplicity, assume a single asset class "ALL"
    asset_classes = ["ALL"]
    average_metrics = {}
    list_metrics = {}

    all_results = {"ALL": {}}
    for bp in BACKTEST_AVERAGE_BASIS_POINTS:
        bp_suffix = _basis_point_suffix(bp)
        for m in _metrics + _rescaled_metrics:
            key = m + bp_suffix
            all_results["ALL"][key] = []
        all_results["ALL"][f"sharpe_ratio_years{bp_suffix}"] = []
    
    # Loop through experiment directories (each corresponds to a train interval)
    for exp_dirs in experiment_directories.values():
        for exp_dir in exp_dirs:
            metrics_csv = os.path.join(exp_dir, "metrics.csv")
            if not os.path.exists(metrics_csv):
                continue
            df = pd.read_csv(metrics_csv)
            for col in df.columns:
                if col in all_results["ALL"]:
                    all_results["ALL"][col].append(df[col].iloc[0])
            # Yearly sharpe ratios can be added here if stored per window.
    
    # Aggregate (compute means) for each metric
    window_history = copy.deepcopy(all_results["ALL"])
    aggregated = {}
    for key, values in all_results["ALL"].items():
        aggregated[key] = np.mean(values) if values else None
    for bp in BACKTEST_AVERAGE_BASIS_POINTS:
        suffix = _basis_point_suffix(bp)
        key = f"sharpe_ratio_years{suffix}"
        aggregated[f"sharpe_ratio_years_std{suffix}"] = np.std(window_history[key]) if window_history[key] else None

    average_metrics["ALL"] = aggregated
    list_metrics["ALL"] = window_history

    # Write aggregated results to JSON files.
    with open(os.path.join(output_directory, "average_results.json"), "w") as file:
        file.write(json.dumps(average_metrics, indent=4))
    with open(os.path.join(output_directory, "list_results.json"), "w") as file:
        file.write(json.dumps(list_metrics, indent=4))
    print("Aggregated metrics saved in", output_directory)

# =============================================================================
# Main Function: Run Classical Methods
# =============================================================================
def scale_position(returns_data, target_vol=0.15):
    """
    Rescale the 'position' column so that the annual volatility performance metric (derived from captured returns)
    equals target_vol.
    
    This function:
      1. Computes the raw performance metrics (which includes 'annual_volatility') via calc_performance_metrics.
      2. Computes the scaling factor as factor = target_vol / raw_annual_volatility.
      3. Multiplies the positions by this factor.
      4. Updates the captured returns accordingly.
      5. Prints debug information showing the raw and new annual volatility.
    
    Assumes calc_performance_metrics(returns_data.set_index("time"), "") returns a dictionary with key 'annual_volatility'.
    """
    # Compute raw performance metrics to extract the annual volatility.
    raw_metrics = calc_performance_metrics(returns_data.set_index("time"), "")
    raw_annual_vol = raw_metrics.get("annual_volatility", None)
    if raw_annual_vol is None:
        raise ValueError("calc_performance_metrics did not return an 'annual_volatility' key.")
    
    # Compute scaling factor based on the annual volatility performance metric.
    factor = target_vol / raw_annual_vol if raw_annual_vol != 0 else 1.0

    # Scale the positions.
    returns_data["position"] *= factor
    # Update captured returns accordingly.
    returns_data["captured_returns"] = returns_data["position"] * returns_data["returns"]
    
    # For debugging, recalculate the performance metrics after scaling.
    new_metrics = calc_performance_metrics(returns_data.set_index("time"), "")
    print(f"[Scaling Debug] Raw annual vol: {raw_annual_vol:.6f}, scaling factor: {factor:.4f}, new annual vol: {new_metrics.get('annual_volatility', np.nan):.6f}")
    
    return returns_data

def run_classical_methods(
    features_file_path,
    train_intervals,
    reference_experiment,
    long_only_experiment_name="long_only",
    short_only_experiment_name="short_only",
    tsmom_experiment_name="tsmom",
    # macd_experiment_name="macd",
):
    """
    Run classical experiments (TSMOM, Long Only, and Short Only) on each train interval.
    
    For each experiment and each train interval, we:
      - Compute performance metrics using the raw (unscaled) signal.
      - Apply volatility scaling based on the annual volatility performance metric (derived from captured returns)
        so that annual volatility becomes 15%.
      - Recompute performance metrics on the scaled signal.
    
    Both raw and scaled metrics are collected as separate rows in the final aggregated DataFrame.
    """
    # Create main directories for experiments.
    for exp in [long_only_experiment_name, short_only_experiment_name, tsmom_experiment_name]:
        os.makedirs(_get_directory_name(exp), exist_ok=True)
    
    # Dictionary to track experiment directories.
    experiment_directories = {}
    for momentum_val in [0, 0.5, 1]:
        experiment_directories[f"{tsmom_experiment_name}_{momentum_val}"] = []
    experiment_directories[long_only_experiment_name] = []
    experiment_directories[short_only_experiment_name] = []
    
    # List to collect metrics dictionaries.
    all_metrics = []
    
    for train_interval in train_intervals:
        # Load features and merge with reference returns.
        raw_data = pd.read_csv(features_file_path, parse_dates=True)
        ref_file_path = os.path.join(
            "results",
            reference_experiment,
            f"{train_interval[1]}-{train_interval[2]}",
            "captured_returns_sw.csv"
        )
        print(f"Processing train interval: {train_interval}")
        if not os.path.exists(ref_file_path):
            print(f"Reference file {ref_file_path} not found. Skipping interval {train_interval}.")
            continue
        
        reference = pd.read_csv(ref_file_path, parse_dates=True)
        base_returns_data = raw_data.merge(
            reference[["time", "identifier", "returns"]],
            left_on=["date", "ticker"],
            right_on=["time", "identifier"],
        )
        
        # ---------------------------
        # TSMOM Experiments
        # ---------------------------
        for momentum_val in [0, 0.5, 1]:
            experiment_name = f"{tsmom_experiment_name}_{momentum_val}"
            dir_path = _get_directory_name(experiment_name, train_interval)
            os.makedirs(dir_path, exist_ok=True)
            experiment_directories[experiment_name].append(dir_path)
            
            # --- Raw Signal ---
            data_raw = base_returns_data.copy()
            data_raw["position"] = intermediate_momentum_position(momentum_val, data_raw)
            data_raw["captured_returns"] = data_raw["position"] * data_raw["returns"]
            data_raw = data_raw.reset_index(drop=True)
            
            # Save raw captured returns.
            raw_file = os.path.join(dir_path, "captured_returns_raw.csv")
            data_raw.to_csv(raw_file, index=False)
            print(f"Saved raw captured returns to: {raw_file}")
            
            raw_metrics = calc_performance_metrics(data_raw.set_index("time"), "")
            raw_metrics.update({
                "experiment": experiment_name,
                "train_interval": str(train_interval),
                "scaling": "raw"
            })
            all_metrics.append(raw_metrics)
            
            # --- Scaled Signal ---
            data_scaled = base_returns_data.copy()
            data_scaled["position"] = intermediate_momentum_position(momentum_val, data_scaled)
            # Apply scaling based on the annual volatility performance metric.
            data_scaled = scale_position(data_scaled, target_vol=0.15)
            data_scaled = data_scaled.reset_index(drop=True)
            
            scaled_file = os.path.join(dir_path, "captured_returns_scaled.csv")
            data_scaled.to_csv(scaled_file, index=False)
            print(f"Saved scaled captured returns to: {scaled_file}")
            
            scaled_metrics = calc_performance_metrics(data_scaled.set_index("time"), "")
            scaled_metrics.update({
                "experiment": experiment_name,
                "train_interval": str(train_interval),
                "scaling": "15% annualized"
            })
            all_metrics.append(scaled_metrics)
        
        # ---------------------------
        # Long Only Experiment
        # ---------------------------
        experiment_name = long_only_experiment_name
        dir_path = _get_directory_name(experiment_name, train_interval)
        os.makedirs(dir_path, exist_ok=True)
        experiment_directories[experiment_name].append(dir_path)
        
        # --- Raw Signal ---
        data_raw = base_returns_data.copy()
        data_raw["position"] = 1.0
        data_raw["captured_returns"] = data_raw["position"] * data_raw["returns"]
        data_raw = data_raw.reset_index(drop=True)
        
        raw_file = os.path.join(dir_path, "captured_returns_raw.csv")
        data_raw.to_csv(raw_file, index=False)
        print(f"Saved raw captured returns to: {raw_file}")
        
        raw_metrics = calc_performance_metrics(data_raw.set_index("time"), "")
        raw_metrics.update({
            "experiment": experiment_name,
            "train_interval": str(train_interval),
            "scaling": "raw"
        })
        all_metrics.append(raw_metrics)
        
        # --- Scaled Signal ---
        data_scaled = base_returns_data.copy()
        data_scaled["position"] = 1.0
        data_scaled = scale_position(data_scaled, target_vol=0.15)
        data_scaled = data_scaled.reset_index(drop=True)
        
        scaled_file = os.path.join(dir_path, "captured_returns_scaled.csv")
        data_scaled.to_csv(scaled_file, index=False)
        print(f"Saved scaled captured returns to: {scaled_file}")
        
        scaled_metrics = calc_performance_metrics(data_scaled.set_index("time"), "")
        scaled_metrics.update({
            "experiment": experiment_name,
            "train_interval": str(train_interval),
            "scaling": "15% annualized"
        })
        all_metrics.append(scaled_metrics)
        
        # ---------------------------
        # Short Only Experiment
        # ---------------------------
        experiment_name = short_only_experiment_name
        dir_path = _get_directory_name(experiment_name, train_interval)
        os.makedirs(dir_path, exist_ok=True)
        experiment_directories[experiment_name].append(dir_path)
        
        # --- Raw Signal ---
        data_raw = base_returns_data.copy()
        data_raw["position"] = -1.0
        data_raw["captured_returns"] = data_raw["position"] * data_raw["returns"]
        data_raw = data_raw.reset_index(drop=True)
        
        raw_file = os.path.join(dir_path, "captured_returns_raw.csv")
        data_raw.to_csv(raw_file, index=False)
        print(f"Saved raw captured returns to: {raw_file}")
        
        raw_metrics = calc_performance_metrics(data_raw.set_index("time"), "")
        raw_metrics.update({
            "experiment": experiment_name,
            "train_interval": str(train_interval),
            "scaling": "raw"
        })
        all_metrics.append(raw_metrics)
        
        # --- Scaled Signal ---
        data_scaled = base_returns_data.copy()
        data_scaled["position"] = -1.0
        data_scaled = scale_position(data_scaled, target_vol=0.15)
        data_scaled = data_scaled.reset_index(drop=True)
        
        scaled_file = os.path.join(dir_path, "captured_returns_scaled.csv")
        data_scaled.to_csv(scaled_file, index=False)
        print(f"Saved scaled captured returns to: {scaled_file}")
        
        scaled_metrics = calc_performance_metrics(data_scaled.set_index("time"), "")
        scaled_metrics.update({
            "experiment": experiment_name,
            "train_interval": str(train_interval),
            "scaling": "15% annualized"
        })
        all_metrics.append(scaled_metrics)
    
    # Optionally, aggregate strategy metrics across intervals.
    output_dir = "aggregated_results"
    os.makedirs(output_dir, exist_ok=True)
    aggregate_strategy_metrics(
        experiment_name=tsmom_experiment_name,  # or a composite name if desired.
        train_intervals=train_intervals,
        experiment_directories=experiment_directories,
        output_directory=output_dir,
        asset_class_dictionary=None,
        standard_window_size=6
    )
    
    # Combine all metrics into a DataFrame and print.
    strategy_metrics_df = pd.DataFrame(all_metrics)
    print("\nAggregated Strategy Metrics (Raw and Scaled):")
    print(strategy_metrics_df, flush=True)
    
    return strategy_metrics_df