import pandas as pd
import matplotlib.pyplot as plt

# Define file paths for each strategy.
# path_experiment = r"C:\Users\Sean\Documents\gml-master\results\experiment_gml_TEST_lstm_cpnone_len63_notime_div_v1\2021-2022\captured_returns_sw.csv"
path_long_only_sw  = r"C:\Users\Sean\Documents\gml-master\results\long_only\2021-2023\captured_returns_sw.csv"
path_long_only_sw = r"C:\Users\Sean\Documents\gml-master\results\long_only\2021-2023\captured_returns_sw.csv"

# Read the CSV files.
df_long_only_sw  = pd.read_csv(path_long_only_sw, parse_dates=["time"])
# df_long_only_      = pd.read_csv(path_tsmom, parse_dates=["time"])

def compute_cumulative_returns(df, return_col="captured_returns"):
    """
    Groups the DataFrame by date (time), computes the average captured return
    for each day, and then calculates the cumulative (geometric) return.
    
    Cumulative return is computed as:
      cumulative_return(t) = Π₍ᵢ₌₁₎ᵗ (1 + avg_return(i)) - 1
    """
    # Group by date and average the captured returns.
    df_grouped = df.groupby("time")[return_col].mean().reset_index(name="avg_captured_returns")
    # Sort by date.
    df_grouped = df_grouped.sort_values("time")
    # Compute the cumulative return.
    df_grouped["cumulative_returns"] = (1 + df_grouped["avg_captured_returns"]).cumprod() - 1
    return df_grouped

# Compute cumulative returns for each strategy.
# exp_returns      = compute_cumulative_returns(df_experiment)
long_only_returns = compute_cumulative_returns(df_long_only_sw)
# tsmom_returns    = compute_cumulative_returns(df_tsmom)

# Plot the cumulative returns.
plt.figure(figsize=(10, 6))
plt.plot(exp_returns["time"], exp_returns["cumulative_returns"], label="Experiment Strategy", lw=2)
plt.plot(long_only_returns["time"], long_only_returns["cumulative_returns"], label="Long Only Strategy", lw=2)
plt.plot(tsmom_returns["time"], tsmom_returns["cumulative_returns"], label="TSMOM Strategy", lw=2)

plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Returns Comparison")
# plt.yscale("log")  # Use a log scale on the y-axis
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
