import pandas as pd

df = pd.read_csv("data/straddle_features/features.csv")

# Group by the "ticker" column and count the rows per ticker
rows_per_ticker = df.groupby("ticker").size().reset_index(name='row_count')

print(rows_per_ticker[:50])

print(rows_per_ticker[50:])