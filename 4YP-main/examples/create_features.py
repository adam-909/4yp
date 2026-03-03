import argparse
import datetime as dt
from typing import List
import numpy as np

import pandas as pd

from data.pull_data import pull_straddle_sample_data, pull_equities_sample_data
from settings.default import (
    ALL_TICKERS,
    FEATURES_STRADDLES_FILE_PATH,
    FEATURES_EQUITIES_FILE_PATH,
)
from gml.data_prep import (
    deep_momentum_strategy_features,
)

from settings.default import STRADDLE
import pandas as pd

from settings.default import STITCH_STRADDLE_DATA



# This function is fucking me for some reason 


def fill_missing_with_ema(df: pd.DataFrame, date_col: str, ticker_col: str, span: int) -> pd.DataFrame:
    # Ensure the date column is in datetime format.
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Compute the union of all dates across tickers.
    union_dates = pd.DatetimeIndex(sorted(df[date_col].unique()))
    
    filled_dfs = []
    
    # Process each ticker group separately.
    for ticker, group in df.groupby(ticker_col):
        # Set the date column as the index and reindex to the union of all dates.
        group = group.set_index(date_col)
        group = group.reindex(union_dates)
        
        # Ensure the ticker column is present for all rows.
        group[ticker_col] = ticker
        
        # For non-numeric columns, fill missing values using forward/backward fill.
        for col in group.columns:
            if col == ticker_col:
                continue
            if not np.issubdtype(group[col].dtype, np.number):
                group[col] = group[col].ffill().bfill()
        
        # Identify numeric columns.
        numeric_cols = group.select_dtypes(include=[np.number]).columns
        
        # 1) Fill missing values in each numeric column using an EMA over the full time series.
        for col in numeric_cols:
            ema_series = group[col].ewm(span=span, adjust=False, min_periods=1).mean()
            group[col] = group[col].fillna(ema_series)
        
        # 2) Replace outliers in each numeric column:
        #    If a value is more than 5 std from the mean, replace it with
        #    the EMA of the previous 5 days (shifted by 1).
        for col in numeric_cols:
            mean_val = group[col].mean()
            std_val = group[col].std()
            if std_val == 0 or np.isnan(std_val):
                # If std is zero or NaN (e.g. all identical values), skip outlier logic.
                continue
            
            outlier_mask = (group[col] - mean_val).abs() > 5 * std_val
            
            # Compute the EMA over the previous 5 days for this column.
            ema_prev5 = group[col].ewm(span=5, adjust=False, min_periods=1).mean().shift(1)
            # Fill any leading NaNs in ema_prev5 by forward fill.
            ema_prev5.fillna(method='ffill', inplace=True)
            
            # Replace outlier values with the previous-5-day EMA.
            group.loc[outlier_mask, col] = ema_prev5[outlier_mask]
        
        filled_dfs.append(group)
    
    # Combine all ticker groups into one DataFrame.
    result_df = pd.concat(filled_dfs)
    
    # Recompute date-derived features from the index.
    if isinstance(result_df.index, pd.DatetimeIndex):
        result_df["day_of_week"] = result_df.index.dayofweek
        result_df["day_of_month"] = result_df.index.day
        result_df["month_of_year"] = result_df.index.month
        result_df["year"] = result_df.index.year
        result_df["date"] = result_df.index  # keep a copy of the date as a column
    
    # Sort by ticker and then by date.
    result_df = result_df.sort_values(by=[ticker_col, "date"])
    
    return result_df

# def fill_missing_with_ema(df: pd.DataFrame, date_col: str, ticker_col: str, span: int) -> pd.DataFrame:
#     # Ensure the date column is in datetime format.
#     df[date_col] = pd.to_datetime(df[date_col])
    
#     # Compute the union of all dates across tickers.
#     union_dates = pd.DatetimeIndex(sorted(df[date_col].unique()))
    
#     filled_dfs = []
    
#     # Process each ticker group separately.
#     for ticker, group in df.groupby(ticker_col):
#         # Set the date column as the index.
#         group = group.set_index(date_col)
#         # Reindex the group to have all dates in the union.
#         group = group.reindex(union_dates)
        
#         # Ensure the ticker column is present for all rows.
#         group[ticker_col] = ticker
        
#         # For non-numeric columns (e.g., exdate, moneyness if not numeric, etc.), fill using forward and backward fill.
#         for col in group.columns:
#             # Skip ticker column (already filled).
#             if col == ticker_col:
#                 continue
#             # Check if the column is non-numeric.
#             if not np.issubdtype(group[col].dtype, np.number):
#                 group[col] = group[col].ffill().bfill()
        
#         # For numeric columns, fill missing values using the EMA computed over the full time series.
#         numeric_cols = group.select_dtypes(include=[np.number]).columns
#         for col in numeric_cols:
#             ema_series = group[col].ewm(span=span, adjust=False, min_periods=1).mean()
#             group[col] = group[col].fillna(ema_series)
        
#         filled_dfs.append(group)
    
#     # Combine all ticker groups.
#     result_df = pd.concat(filled_dfs)
    
#     # Recompute the date-derived features from the index.
#     if isinstance(result_df.index, pd.DatetimeIndex):
#         result_df["day_of_week"] = result_df.index.dayofweek
#         result_df["day_of_month"] = result_df.index.day
#         result_df["month_of_year"] = result_df.index.month
#         result_df["year"] = result_df.index.year
#         result_df["date"] = result_df.index  # keep a copy of the date as a column
    
#     # Finally, sort by ticker and then by date.
#     result_df = result_df.sort_values(by=[ticker_col, "date"])
    
#     return result_df


def main(
    tickers: List[str],
    output_file_path: str,
) -> None:

    if STRADDLE:
        features = pd.concat(
            [
                deep_momentum_strategy_features(
                    pull_straddle_sample_data(ticker)
                ).assign(ticker=ticker)
                for ticker in tickers
            ]
        )
        
        # if not STITCH_STRADDLE_DATA:
        features = fill_missing_with_ema(features, date_col="date", ticker_col="ticker", span=5)



    else:
        features = pd.concat(
            [
                deep_momentum_strategy_features(
                    pull_equities_sample_data(ticker)
                ).assign(ticker=ticker)
                for ticker in tickers
            ]
        )

    
        
    if "stitched_price" in features.columns:
        features = features.drop(columns=["stitched_price"])
        
    if "srs" in features.columns:
        features = features.drop(columns=["srs"])   
         
        
    features = features.applymap(
        lambda x: np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10) if isinstance(x, (int, float)) else x
    )
    
    features.date = features.index
    features.index.name = "Date"
    features.to_csv(output_file_path)


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
        parser.add_argument(
            "--output_file_path",
            "-o",
            type=str,
            default=(
                FEATURES_STRADDLES_FILE_PATH
                if STRADDLE
                else FEATURES_EQUITIES_FILE_PATH
            ),
            help="Output file path for the CSV. Defaults to FEATURES_EQUITIES_FILE_PATH.",
        )

        args = parser.parse_known_args()[0]

        return args.tickers, args.output_file_path

    main(*get_args())
