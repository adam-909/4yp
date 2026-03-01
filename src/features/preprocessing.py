"""
Feature preprocessing (winsorization).
"""

import pandas as pd


def winsorize_ewm(
    data: pd.DataFrame,
    half_life: int = 252,
    n_std: float = 5.0
) -> pd.DataFrame:
    """
    Winsorize data using exponentially weighted mean and std.

    Applies floor/cap at mu_EWM +/- n_std * sigma_EWM.

    Args:
        data: Input DataFrame
        half_life: Half-life for EWM calculation (in days)
        n_std: Number of standard deviations for floor/cap

    Returns:
        Winsorized DataFrame
    """
    ewm_mean = data.ewm(halflife=half_life, min_periods=1).mean()
    ewm_std = data.ewm(halflife=half_life, min_periods=1).std()

    lower = ewm_mean - n_std * ewm_std
    upper = ewm_mean + n_std * ewm_std

    winsorized = data.clip(lower=lower, upper=upper)

    return winsorized
