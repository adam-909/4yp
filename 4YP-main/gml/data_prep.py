import os

import numpy as np
import pandas as pd


from gml.classical_strategies import (
    MACDStrategy,
    calc_returns,
    calc_daily_vol,
    calc_vol_scaled_returns,
)

from settings.default import STRADDLE

VOL_THRESHOLD = 5  # multiple to winsorise by
HALFLIFE_WINSORISE = 252

from settings.default import STITCH_STRADDLE_DATA


def deep_momentum_strategy_features(df_asset: pd.DataFrame) -> pd.DataFrame:
    """prepare input features for deep learning model

    Args:
        df_asset (pd.DataFrame): time-series for asset with column close

    Returns:
        pd.DataFrame: input features
    """

    if STRADDLE:
        if STITCH_STRADDLE_DATA:
            df_asset = df_asset[
                ~df_asset["stitched_price"].isna()
                | ~df_asset["stitched_price"].isnull()
                | (df_asset["stitched_price"] > 1e-20)  # price is zero
            ].copy()
            df_asset["srs"] = df_asset["stitched_price"]
        else:
            df_asset = df_asset[
                ~df_asset["straddle_price"].isna()
                | ~df_asset["straddle_price"].isnull()
                | (df_asset["straddle_price"] > 1e-8)  # price is zero
            ].copy()
            df_asset["srs"] = df_asset["straddle_price"]

    else:
        df_asset = df_asset[
            ~df_asset["price"].isna()
            | ~df_asset["price"].isnull()
            | (df_asset["price"] > 1e-8)  # price is zero
        ].copy()
        df_asset["srs"] = df_asset["price"]

    # winsorize using rolling 5X standard deviations to remove outliers
    ewm = df_asset["srs"].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    df_asset["srs"] = np.minimum(df_asset["srs"], means + VOL_THRESHOLD * stds)
    df_asset["srs"] = np.maximum(df_asset["srs"], means - VOL_THRESHOLD * stds)

    # TODO add logic to amend for rolling contracts

    df_asset["daily_returns"] = calc_returns(df_asset["srs"])
    df_asset["daily_vol"] = calc_daily_vol(df_asset["daily_returns"])
    # vol scaling and shift to be next day returns
    df_asset["target_returns"] = calc_vol_scaled_returns(
        df_asset["daily_returns"], df_asset["daily_vol"]
    ).shift(-1)

    def calc_normalised_returns(day_offset):
        return (
            calc_returns(df_asset["srs"], day_offset)
            / df_asset["daily_vol"]
            / np.sqrt(day_offset)
        )

    # Using different lookback periods for normalised returns
    df_asset["norm_daily_return"] = calc_normalised_returns(1)   # daily
    df_asset["norm_monthly_return"] = calc_normalised_returns(5)   # weekly
    df_asset["norm_quarterly_return"] = calc_normalised_returns(10)  # biweekly
    df_asset["norm_biannual_return"] = calc_normalised_returns(15)  # triweekly
    df_asset["norm_annual_return"] = calc_normalised_returns(20)    # monthly

    # Compute trend features using MACD signal
    trend_combinations = [(2, 8), (4, 16), (8, 32)]
    for short_window, long_window in trend_combinations:
        df_asset[f"macd_{short_window}_{long_window}"] = MACDStrategy.calc_signal(
            df_asset["srs"], short_window, long_window
        )

    # Date features
    if len(df_asset):
        # Ensure the index is datetime
        if not isinstance(df_asset.index, pd.DatetimeIndex):
            df_asset.index = pd.to_datetime(df_asset.index)
        df_asset["day_of_week"] = df_asset.index.dayofweek
        df_asset["day_of_month"] = df_asset.index.day
        # df_asset["week_of_year"] = df_asset.index.weekofyear  # deprecated in newer pandas versions
        df_asset["month_of_year"] = df_asset.index.month
        df_asset["year"] = df_asset.index.year
        df_asset["date"] = df_asset.index  # duplication but sometimes makes life easier
    else:
        df_asset["day_of_week"] = []
        df_asset["day_of_month"] = []
        # df_asset["week_of_year"] = []
        df_asset["month_of_year"] = []
        df_asset["year"] = []
        df_asset["date"] = []

    # --- Added features ---
    # Calculate log-moneyness if the 'moneyness' column exists.
    if "moneyness" in df_asset.columns:
        df_asset["log_moneyness"] = np.log(df_asset["moneyness"])
    else:
        # Optionally, handle missing moneyness (e.g., set to NaN or compute from available data)
        df_asset["log_moneyness"] = np.nan

    # Calculate time to expiry in years, using the 'exdate' column.
    # We assume that the index of df_asset represents the trade date.
    if "exdate" in df_asset.columns:
        # Ensure exdate is datetime
        df_asset["exdate"] = pd.to_datetime(df_asset["exdate"])
        # Calculate the difference in days and convert to years (using 365.25 days per year)
        df_asset["time_to_expiry"] = (df_asset["exdate"] - df_asset.index).dt.days / 365.25
    else:
        df_asset["time_to_expiry"] = np.nan
        
        
    # df_asset.dropna()
    # print(df_asset.tail(5))
    # df_asset = fill_missing_with_ema(df_asset, date_col="date", ticker_col="ticker", span=5)

    # return df_asset
    return df_asset.dropna()