import pandas as pd
import matplotlib.pyplot as plt

STRADDLE = True

if STRADDLE:
    # Define file paths for each strategy.
    path_experiment = r"C:\Users\Sean\Documents\gml-master\results\experiment_gml_TEST_lstm_cpnone_len63_notime_div_v1\2021-2022\captured_returns_sw.csv"
    path_long_only = r"C:\Users\Sean\Documents\gml-master\results\long_only\2021-2022\captured_returns_sw.csv"
    path_tsmom = r"C:\Users\Sean\Documents\gml-master\results\tsmom\2021-2022\captured_returns_sw.csv"
else:
    # Define file paths for each strategy.
    path_experiment = r"C:\Users\Sean\Documents\gml-master\results\straddles\experiment_gml_lstm_cpnone_len63_notime_div_v1\2021-2023\captured_returns_sw.csv"
    path_long_only = r"C:\Users\Sean\Documents\gml-master\results\straddles\long_only\2021-2023\captured_returns_sw.csv"
    path_tsmom = r"C:\Users\Sean\Documents\gml-master\results\straddles\tsmom\2021-2023\captured_returns_sw.csv"

# Read the CSV files.
df_experiment = pd.read_csv(path_experiment, parse_dates=["time"])
# df_long_only = pd.read_csv(path_long_only, parse_dates=["time"])
# df_tsmom = pd.read_csv(path_tsmom, parse_dates=["time"])

print(df_experiment.head())

def compute_cumulative_returns(df, return_col="captured_returns"):
    """
    Groups the DataFrame by date (time), computes the average captured return
    for each day based on the specified column, and then calculates the cumulative (geometric) return.
    
    Cumulative return is computed as:
      cumulative_return(t) = Π₍ᵢ₌₁₎ᵗ (1 + avg_return(i)) - 1
    """
    # Group by date and average the selected returns.
    df_grouped = (
        df.groupby("time")[return_col]
          .mean()
          .reset_index(name="avg_captured_returns")
    )
    # Sort by date.
    df_grouped = df_grouped.sort_values("time")
    # Compute the cumulative return.
    df_grouped["cumulative_returns"] = (1 + df_grouped["avg_captured_returns"]).cumprod() - 1
    return df_grouped

# Define the transaction cost columns and their corresponding labels.
# The keys are the column names in the CSV, and the values are the labels for the plot.
tc_columns = {
    "captured_returns": "Baseline",
    "captured_returns_5_0_bps": "5.0 bps",
    "captured_returns_10_0_bps": "10.0 bps",
    "captured_returns_15_0_bps": "15.0 bps",
    "captured_returns_29_0_bps": "29.0 bps",
    "captured_returns_25_0_bps": "25.0 bps",
    "captured_returns_30_0_bps": "30.0 bps"
}

# Create subplots: one for each strategy.
fig, axes = plt.subplots(nrows=3, figsize=(10, 18), sharex=True)

# List of (DataFrame, Strategy Label) tuples.
strategy_data = [
    (df_experiment, "Experiment Strategy"),
    # (df_long_only, "Long Only Strategy"),
    # (df_tsmom, "TSMOM Strategy")
]

for ax, (df, strategy_label) in zip(axes, strategy_data):
    # Plot each transaction cost variant.
    for col, label in tc_columns.items():
        cum_returns = compute_cumulative_returns(df, return_col=col)
        ax.plot(cum_returns["time"], cum_returns["cumulative_returns"], lw=2, label=label)
    ax.set_title(strategy_label)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()
