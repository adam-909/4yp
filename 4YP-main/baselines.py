import pandas as pd
from empyrical import annual_return, annual_volatility, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio

# Load features
df = pd.read_csv('4YP-main/data/straddle_features/features.csv', parse_dates=['date'])
df['date'] = pd.to_datetime(df['date'])

# Filter to 2017-2023
df_test = df[(df['date'] >= '2017-01-01') & (df['date'] <= '2023-12-31')].copy()

# Long only: position = 1.0 always
df_test['position'] = 1.0
df_test['captured_returns'] = df_test['position'] * df_test['daily_returns']

# Aggregate by date
daily_returns = df_test.groupby('date')['captured_returns'].sum() / 88

# Calculate metrics
print("Long Only Strategy (2017-2023):")
print(f"Annual Return: {annual_return(daily_returns):.4f}")
print(f"Annual Vol: {annual_volatility(daily_returns):.4f}")
print(f"Sharpe: {sharpe_ratio(daily_returns):.4f}")
print(f"Max Drawdown: {-max_drawdown(daily_returns):.4f}")
print(f"Sortino: {sortino_ratio(daily_returns):.4f}")
print(f"Calmar: {calmar_ratio(daily_returns):.4f}")