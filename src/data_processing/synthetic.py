"""
Synthetic data generation for testing.

Creates realistic synthetic options and equity data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

from .portfolio import get_portfolio_formation_days


def create_synthetic_options_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic options and equity data for testing.

    Generates realistic option prices and greeks based on simplified
    Black-Scholes assumptions with stochastic volatility.

    Args:
        tickers: List of ticker symbols
        start_date: Start date string
        end_date: End date string
        seed: Random seed for reproducibility

    Returns:
        Tuple of (options_df, equity_df)
    """
    np.random.seed(seed)

    trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(trading_days)
    n_tickers = len(tickers)

    # Generate correlated equity returns with sector structure
    base_corr = 0.4
    sector_corr = 0.7
    n_sectors = 6
    sector_size = max(n_tickers // n_sectors, 1)

    corr_matrix = np.full((n_tickers, n_tickers), base_corr)
    for s in range(n_sectors):
        start_idx = s * sector_size
        end_idx = min((s + 1) * sector_size, n_tickers)
        corr_matrix[start_idx:end_idx, start_idx:end_idx] = sector_corr
    np.fill_diagonal(corr_matrix, 1.0)

    L = np.linalg.cholesky(corr_matrix)

    base_vol = 0.02
    returns = np.zeros((n_days, n_tickers))
    vol_process = np.ones(n_tickers) * base_vol

    for t in range(n_days):
        vol_shock = np.random.normal(0, 0.001, n_tickers)
        vol_process = 0.98 * vol_process + 0.02 * base_vol + vol_shock
        vol_process = np.clip(vol_process, 0.005, 0.08)

        z = np.random.normal(0, 1, n_tickers)
        corr_z = L @ z
        returns[t] = corr_z * vol_process

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    equity_df = pd.DataFrame(prices, index=trading_days, columns=tickers)

    # Generate options data
    options_records = []
    pfds = get_portfolio_formation_days(start_date, end_date)

    from scipy.stats import norm

    for pfd in pfds:
        if pfd not in trading_days:
            continue

        exdate = pfd + pd.Timedelta(days=30)
        days_to_friday = (4 - exdate.weekday()) % 7
        exdate = exdate + pd.Timedelta(days=days_to_friday)

        pfd_idx = trading_days.get_loc(pfd)

        for ticker_idx, ticker in enumerate(tickers):
            spot = equity_df.loc[pfd, ticker]
            strikes = spot * np.array([0.95, 0.975, 1.0, 1.025, 1.05])

            for strike in strikes:
                T = 30 / 252
                sigma = vol_process[ticker_idx] * np.sqrt(252)
                r = 0.02

                moneyness = spot / strike
                spread_pct = 0.02 + 0.03 * abs(moneyness - 1)

                option_days = trading_days[(trading_days >= pfd) & (trading_days <= exdate)]

                for day in option_days:
                    current_spot = equity_df.loc[day, ticker]
                    T_remaining = max((exdate - day).days / 252, 1/252)

                    d1 = (np.log(current_spot / strike) + (r + 0.5 * sigma**2) * T_remaining) / (sigma * np.sqrt(T_remaining))
                    d2 = d1 - sigma * np.sqrt(T_remaining)

                    call_price = max(current_spot * norm.cdf(d1) - strike * np.exp(-r * T_remaining) * norm.cdf(d2), 0.01)
                    put_price = max(strike * np.exp(-r * T_remaining) * norm.cdf(-d2) - current_spot * norm.cdf(-d1), 0.01)

                    call_delta = norm.cdf(d1)
                    put_delta = norm.cdf(d1) - 1

                    options_records.append({
                        'date': day,
                        'ticker': ticker,
                        'exdate': exdate,
                        'strike': strike,
                        'cp_flag': 'C',
                        'best_bid': call_price * (1 - spread_pct/2),
                        'best_offer': call_price * (1 + spread_pct/2),
                        'delta': call_delta,
                        'impl_volatility': sigma,
                        'open_interest': np.random.randint(100, 10000)
                    })

                    options_records.append({
                        'date': day,
                        'ticker': ticker,
                        'exdate': exdate,
                        'strike': strike,
                        'cp_flag': 'P',
                        'best_bid': put_price * (1 - spread_pct/2),
                        'best_offer': put_price * (1 + spread_pct/2),
                        'delta': put_delta,
                        'impl_volatility': sigma,
                        'open_interest': np.random.randint(100, 10000)
                    })

    options_df = pd.DataFrame(options_records)

    return options_df, equity_df
