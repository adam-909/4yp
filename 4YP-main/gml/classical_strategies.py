import numpy as np
import pandas as pd

from typing import Dict, List, Tuple

from empyrical import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    downside_risk,
    annual_return,
    annual_volatility,
    # cum_returns,
)

from settings.default import STRADDLE

VOL_LOOKBACK = 20  # for ex-ante volatility
VOL_TARGET = 0.15  # 15% volatility target

from settings.default import STITCH_STRADDLE_DATA

def calc_performance_metrics(
    data: pd.DataFrame, metric_suffix="", num_identifiers=None
) -> dict:
    """Performance metrics for evaluating strategy

    Args:
        captured_returns (pd.DataFrame): dataframe containing captured returns, indexed by date

    Returns:
        dict: dictionary of performance metrics
    """
    if not num_identifiers:
        num_identifiers = len(data.dropna()["identifier"].unique())
    srs = data.dropna().groupby(level=0)["captured_returns"].sum() / num_identifiers
    return {
        f"annual_return{metric_suffix}": annual_return(srs),
        f"annual_volatility{metric_suffix}": annual_volatility(srs),
        f"sharpe_ratio{metric_suffix}": sharpe_ratio(srs),
        f"downside_risk{metric_suffix}": downside_risk(srs),
        f"sortino_ratio{metric_suffix}": sortino_ratio(srs),
        f"max_drawdown{metric_suffix}": -max_drawdown(srs),
        f"calmar_ratio{metric_suffix}": calmar_ratio(srs),
        f"perc_pos_return{metric_suffix}": len(srs[srs > 0.0]) / len(srs),
        f"profit_loss_ratio{metric_suffix}": np.mean(srs[srs > 0.0])
        / np.mean(np.abs(srs[srs < 0.0])),
    }


def calc_performance_metrics_subset(srs: pd.Series, metric_suffix="") -> dict:
    """Performance metrics for evaluating strategy

    Args:
        captured_returns (pd.Series): series containing captured returns, aggregated by date

    Returns:
        dict: dictionary of performance metrics
    """
    return {
        f"annual_return{metric_suffix}": annual_return(srs),
        f"annual_volatility{metric_suffix}": annual_volatility(srs),
        f"downside_risk{metric_suffix}": downside_risk(srs),
        f"max_drawdown{metric_suffix}": -max_drawdown(srs),
    }


# TODO: adjust for rollovers?
def calc_returns(srs: pd.Series, day_offset: int = 1) -> pd.Series:
    """for each element of a pandas time-series srs,
    calculates the returns over the past number of days
    specified by offset

    Args:
        srs (pd.Series): time-series of prices
        day_offset (int, optional): number of days to calculate returns over. Defaults to 1.

    Returns:
        pd.Series: series of returns
    """

    if STRADDLE and not STITCH_STRADDLE_DATA:
            srs.index = pd.to_datetime(srs.index)

            # Calculate standard returns
            returns = srs / srs.shift(day_offset) - 1.0
            returns = returns.sort_index()

            # Identify second Mondays of each month by grouping on year and month.
            second_mondays = []
            grouped = srs.groupby([srs.index.year, srs.index.month])
            for _, group in grouped:
                # Ensure the group's index is in datetime format
                dti = pd.to_datetime(group.index)
                # Find all dates that are Mondays (weekday 0)
                mondays = dti[dti.weekday == 0]
                if len(mondays) > 1:
                    second_mondays.append(mondays[1])

            # Replace returns on second Mondays with the 5-day EMA of previous returns.
            for date in second_mondays:
                if date in returns.index:
                    # Use get_indexer_for to safely get the integer location
                    pos = returns.index.get_indexer_for([date])[0]
                    if pos >= 5:  # Ensure we have 5 days of history
                        window = returns.iloc[pos - 5 : pos]
                        if len(window) == 5:
                            ema_value = window.ewm(span=5, adjust=False).mean().iloc[-1]
                            returns.iloc[pos] = ema_value
    else:
        returns = srs / srs.shift(day_offset) - 1.0

    return returns


# def calc_returns(srs: pd.Series, day_offset: int = 1) -> pd.Series:
#     """
#     Calculates returns over the past number of days specified by day_offset.
#     On the second Monday of each month, replaces the return with the EMA
#     (5-day window) of the previous 5 days.

#     Args:
#         srs (pd.Series): Time-series of prices with a DatetimeIndex.
#         day_offset (int, optional): Number of days to calculate returns over. Defaults to 1.

#     Returns:
#         pd.Series: Series of returns with EMA-adjusted values on second Mondays.
#     """
#     # Ensure the index is a DatetimeIndex
#     srs.index = pd.to_datetime(srs.index)

#     # Calculate standard returns
#     returns = srs / srs.shift(day_offset) - 1.0
#     returns = returns.sort_index()

#     # Identify second Mondays of each month by grouping on year and month.
#     second_mondays = []
#     grouped = srs.groupby([srs.index.year, srs.index.month])
#     for _, group in grouped:
#         # Ensure the group's index is in datetime format
#         dti = pd.to_datetime(group.index)
#         # Find all dates that are Mondays (weekday 0)
#         mondays = dti[dti.weekday == 0]
#         if len(mondays) > 1:
#             second_mondays.append(mondays[1])

#     # Replace returns on second Mondays with the 5-day EMA of previous returns.
#     for date in second_mondays:
#         if date in returns.index:
#             # Use get_indexer_for to safely get the integer location
#             pos = returns.index.get_indexer_for([date])[0]
#             if pos >= 5:  # Ensure we have 5 days of history
#                 window = returns.iloc[pos - 5 : pos]
#                 if len(window) == 5:
#                     ema_value = window.ewm(span=5, adjust=False).mean().iloc[-1]
#                     returns.iloc[pos] = ema_value

#     return returns


def calc_daily_vol(daily_returns):
    return (
        daily_returns.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK)
        .std()
        .fillna(method="bfill")
    )


def calc_sharpe_by_year(data: pd.DataFrame, suffix: str = None) -> dict:
    """Sharpe ratio for each year in dataframe

    Args:
        data (pd.DataFrame): dataframe containing captured returns, indexed by date

    Returns:
        dict: dictionary of Sharpe by year
    """
    if not suffix:
        suffix = ""

    data = data.copy()
    data["year"] = data.index.year

    # mean of year is year for same date
    sharpes = (
        data.dropna()[["year", "captured_returns"]]
        .groupby(level=0)
        .mean()
        .groupby("year")
        .apply(lambda y: sharpe_ratio(y["captured_returns"]))
    )

    sharpes.index = "sharpe_ratio_" + sharpes.index.map(int).map(str) + suffix

    return sharpes.to_dict()


def calc_vol_scaled_returns(daily_returns, daily_vol=pd.Series(None)):
    """calculates volatility scaled returns for annualised VOL_TARGET of 15%
    with input of pandas series daily_returns"""
    if not len(daily_vol):
        daily_vol = calc_daily_vol(daily_returns)
    annualised_vol = daily_vol * np.sqrt(252)  # annualised
    return daily_returns * VOL_TARGET / annualised_vol.shift(1)


def calc_net_returns(
    data: pd.DataFrame, list_basis_points: List[float], identifiers=None
):
    if not identifiers:
        identifiers = data["identifier"].unique().tolist()
    cost = np.atleast_2d(list_basis_points) * 1e-4

    dfs = []
    for i in identifiers:
        data_slice = data[data["identifier"] == i].reset_index(drop=True)
        annualised_vol = data_slice["daily_vol"] * np.sqrt(252)
        scaled_position = VOL_TARGET * data_slice["position"] / annualised_vol
        transaction_costs = (
            scaled_position.diff().abs().fillna(0.0).to_frame().to_numpy() * cost
        )  # TODO should probably fill first with initial cost
        net_captured_returns = (
            data_slice[["captured_returns"]].to_numpy() - transaction_costs
        )
        columns = list(
            map(
                lambda c: "captured_returns_" + str(c).replace(".", "_") + "_bps",
                list_basis_points,
            )
        )
        dfs.append(
            pd.concat(
                [data_slice, pd.DataFrame(net_captured_returns, columns=columns)],
                axis=1,
            )
        )
    return pd.concat(dfs).reset_index(drop=True)


class MACDStrategy:
    def __init__(self, trend_combinations: List[Tuple[float, float]] = None):
        """Used to calculated the combined MACD signal for a multiple short/signal combinations,
        as described in https://arxiv.org/pdf/1904.04912.pdf

        Args:
            trend_combinations (List[Tuple[float, float]], optional): short/long trend combinations. Defaults to None.
        """
        if trend_combinations is None:
            self.trend_combinations = [(8, 24), (16, 48), (32, 96)]
        else:
            self.trend_combinations = trend_combinations

    @staticmethod
    def calc_signal(srs: pd.Series, short_timescale: int, long_timescale: int) -> float:
        """Calculate MACD signal for a signal short/long timescale combination

        Args:
            srs ([type]): series of prices
            short_timescale ([type]): short timescale
            long_timescale ([type]): long timescale

        Returns:
            float: MACD signal
        """

        def _calc_halflife(timescale):
            return np.log(0.5) / np.log(1 - 1 / timescale)

        macd = (
            srs.ewm(halflife=_calc_halflife(short_timescale)).mean()
            - srs.ewm(halflife=_calc_halflife(long_timescale)).mean()
        )
        # print(macd)
        # print(macd)
        
        # OLD TIMESPANS
        
        # q = macd / srs.rolling(63).std().fillna(method="bfill")
        # q_out = q / q.rolling(252).std().fillna(method="bfill")
        
        # NEW TIMESPANS
        eps = 1e-8
        rolling_std_5 = srs.rolling(5).std().fillna(method="bfill") + eps
        q = macd / rolling_std_5

        rolling_std_20 = q.rolling(20).std().fillna(method="bfill") + eps
        q_out = q / rolling_std_20
        
        # print(q_out.tail())
        return q_out

    @staticmethod
    def scale_signal(y):
        return y * np.exp(-(y**2) / 4) / 0.89

    def calc_combined_signal(self, srs: pd.Series) -> float:
        """Combined MACD signal

        Args:
            srs (pd.Series): series of prices

        Returns:
            float: MACD combined signal
        """
        return np.sum(
            [self.calc_signal(srs, S, L) for S, L in self.trend_combinations]
        ) / len(self.trend_combinations)
