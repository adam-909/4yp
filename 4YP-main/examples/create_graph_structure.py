import argparse
import datetime as dt
from typing import List
import numpy as np
import pandas as pd
import os

from data.pull_data import pull_straddle_sample_data, pull_equities_sample_data
from settings.default import (
    ALL_TICKERS,
    FEATURES_STRADDLES_FILE_PATH,
    FEATURES_EQUITIES_FILE_PATH,
)
from gml.data_prep import (
    deep_momentum_strategy_features,
)

from settings.default import STRADDLE, PEARSON, GRAPH_NORMALIZATION

GRAPH_THRESHOLD = 0.6

import numpy as np
import pandas as pd
from typing import List

def compute_pearson_log_returns_adjacency(tickers: List[str], data_loader, threshold: float):
    """
    Given a list of tickers, a function to load their data, and a correlation threshold,
    returns:
      1. A DataFrame of log returns for each ticker,
      2. The correlation matrix (Pearson),
      3. The adjacency matrix (binary) based on the given threshold (with no self loops).
    
    Additionally, it computes and displays the ratio of 1's (edges) to the total number of 
    off-diagonal entries in the matrix.
    
    :param tickers: List of ticker symbols.
    :param data_loader: Function that takes a ticker and returns a DataFrame containing
                        at least a 'price' column for that ticker.
    :param threshold: Correlation threshold for creating the adjacency matrix.
    :return: (log_returns_df, corr_matrix, adjacency_matrix)
    """

    # Dictionary to hold log returns per ticker
    log_return_dict = {}

    for ticker in tickers:
        df = data_loader(ticker)

        if "price" not in df.columns:
            raise ValueError(f"No 'price' column found for {ticker}.")

        # Compute log returns: log(price[t]) - log(price[t-1])
        df["log_return"] = np.log(df["price"]).diff()
        log_return_dict[ticker] = df["log_return"]

    # Combine into a single DataFrame
    log_returns_df = pd.DataFrame(log_return_dict)

    # Drop rows that are all NaN (e.g., the first row from diff())
    log_returns_df.dropna(how="all", inplace=True)

    # Compute Pearson correlation matrix
    corr_matrix = log_returns_df.corr(method="pearson")

    # Create adjacency matrix: 1 if |corr| >= threshold, else 0
    adjacency_matrix = (corr_matrix.abs() >= threshold).astype(int)

    # Remove self loops by setting the diagonal to 0
    np.fill_diagonal(adjacency_matrix.values, 0)

    # Compute statistics: total number of 1's divided by the total number of off-diagonal entries.
    n = adjacency_matrix.shape[0]
    total_possible_off_diagonal = n * (n - 1)
    ones_count = adjacency_matrix.values.sum()
    density = ones_count / total_possible_off_diagonal if total_possible_off_diagonal > 0 else 0

    print(f"Graph density (excluding self loops): {ones_count}/{total_possible_off_diagonal} = {density:.4f}")

    return log_returns_df, corr_matrix, adjacency_matrix


def compute_convex_optimization_adjaceny(tickers: List[str], data_loader, threshold: float, feature_data_path: str):
    # TODO implement the convex optimization program
    return None, None, None


def main(
    tickers: List[str],
    # output_file_path: str,
) -> None:
    # Compute adjacency matrix for ALL_TICKERS
    
    if PEARSON:
        _, _, adjacency_matrix = compute_pearson_log_returns_adjacency(
            tickers=ALL_TICKERS,
            data_loader=pull_equities_sample_data,
            threshold=GRAPH_THRESHOLD,
        )
    else:
        _, _, adjacency_matrix = compute_convex_optimization_adjaceny(
            tickers=ALL_TICKERS,
            data_loader=pull_equities_sample_data,
            threshold=GRAPH_THRESHOLD,
            feature_data_path=None,
        )

    # Define output path for adjacency matrix
    if PEARSON:
        graph_file_path = os.path.join("data", "graph_structure", "pearson", f"{GRAPH_THRESHOLD}.csv")
    else:
        graph_file_path = os.path.join("data", "graph_structure", "cvx_opt", f"{GRAPH_THRESHOLD}.csv")  
    
    adjacency_matrix.to_csv(graph_file_path, index=True)
    print(f"Adjacency matrix saved to: {graph_file_path}")

    # Optionally, save the list of tickers to a separate file
    # tickers_file = graph_file_path.replace(".csv", "_tickers.csv")
    # pd.Series(ALL_TICKERS).to_csv(tickers_file, index=False, header=["Ticker"])
    # print(f"List of tickers saved to: {tickers_file}")

if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""
        parser = argparse.ArgumentParser(
            description="Extract deep momentum strategy features for given tickers."
        )
        parser.add_argument(
            "--tickers",
            "-t",
            type=str,
            nargs="*",
            default=ALL_TICKERS,
            help="List of tickers to process. Defaults to ALL_TICKERS.",
        )
        # parser.add_argument(
        #     "--output_file_path",
        #     "-o",
        #     type=str,
        #     default=(
        #         FEATURES_STRADDLES_FILE_PATH
        #         if STRADDLE
        #         else FEATURES_EQUITIES_FILE_PATH
        #     ),
        #     help="Output file path for the CSV. Defaults to FEATURES_EQUITIES_FILE_PATH.",
        # )

        args = parser.parse_known_args()[0]

        return args.tickers
    # , args.output_file_path

    main(get_args())
