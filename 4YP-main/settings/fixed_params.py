MODEL_PARAMS = {
    "architecture": "LSTM",
    "total_time_steps": 20, # old 252
    "early_stopping_patience": 25,
    "multiprocessing_workers": 32,
    "num_epochs": 300,
    "early_stopping_patience": 25,
    "fill_blank_dates": False,
    "split_tickers_individually": True,
    "random_search_iterations": 1 ,
    "evaluate_diversified_val_sharpe": True,
    "train_valid_ratio": 0.80,
    "time_features": False,
    "force_output_sharpe_length": 0,
}

MODEL_PARAMS_GRAPH = {
    "architecture": "GML",
    "total_time_steps": 20,
    "early_stopping_patience": 25,
    "multiprocessing_workers": 32,
    "num_epochs": 300,
    "early_stopping_patience": 25,
    "fill_blank_dates": False,
    "split_tickers_individually": True,
    "random_search_iterations": 1 , # -> 50
    "evaluate_diversified_val_sharpe": True,
    "train_valid_ratio": 0.80,
    "time_features": False,
    "force_output_sharpe_length": 0,
}