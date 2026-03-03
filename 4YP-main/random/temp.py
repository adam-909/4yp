import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Sean\Documents\gml-master\results\exp_lstm_split80_lstm_cpnone_len20_notime_val_v1\2017-2022\captured_returns_sw.csv", parse_dates=["time"])

df_long_only = pd.read_csv(r"C:\Users\Sean\Documents\gml-master\data\straddle_features\features.csv")
# , parse_dates=["time"])


VOL_SCALING = True


def compute_cumulative_returns(df, return_col="captured_returns"):
    """
    Groups the DataFrame by date (time), computes the average captured return
    for each day, and then calculates the cumulative (geometric) return.

    Cumulative return is computed as:
        cumulative_return(t) = Π₍ᵢ₌₁₎ᵗ (1 + avg_return(i)) - 1
    """
    # Group by date and average the captured returns.
    df_grouped = (
        df.groupby("time")[return_col]
        .mean()
        .reset_index(name="avg_captured_returns")
    )
    # Sort by date.
    df_grouped = df_grouped.sort_values("time")
    # Compute the cumulative return.
    df_grouped["cumulative_returns"] = (
        1 + df_grouped["avg_captured_returns"]
    ).cumprod() - 1
    return df_grouped

# Compute cumulative returns for each strategy.
return_column = "captured_returns" if VOL_SCALING else "returns"

# Compute cumulative returns for each strategy.
exp_returns = compute_cumulative_returns(df, return_col=return_column)
print(exp_returns.head())







# Convert the "Date" column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Filter data to the desired period (2017 to 2022).
df_long_only = df[(df['Date'] >= "2017-01-01") & (df['Date'] <= "2022-12-31")]


# --- Compute cumulative returns for the Long Only Strategy ---
# For a long only strategy, we assume a constant 1 position over the entire period and use "target_returns".
long_only_returns = compute_cumulative_returns(df_long_only, return_col="target_returns")
print("Long Only Strategy Cumulative Returns (first few rows):")
print(long_only_returns.head())

# --- Existing Strategy Returns ---
# Assume exp_returns is already computed from your existing strategy.
# If its date column is not in datetime format, convert it. Here we assume it's called "time".
exp_returns['time'] = pd.to_datetime(exp_returns['time'], format='%Y-%m-%d')
# Filter the existing strategy returns to the period 2017-2022.
exp_returns_filtered = exp_returns[(exp_returns['time'] >= "2017-01-01") & (exp_returns['time'] <= "2022-12-31")]

# --- Plotting Both Strategies on the Same Axis ---
plt.figure(figsize=(10, 6))
plt.plot(exp_returns_filtered["time"], exp_returns_filtered["cumulative_returns"],
         label="Existing Strategy", lw=2)
plt.plot(long_only_returns["Date"], long_only_returns["cumulative_returns"],
         label="Long Only Strategy", lw=2)

plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Returns Comparison (2017-2022)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
