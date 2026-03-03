import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

STRADDLE = True
VOL_SCALING = True

# experiment_gml_TEST_lstm_cpnone_len63_notime_div_v1
# "C:\Users\Sean\Documents\gml-master\results\exp_lstm_TEST_split80_lstm_cpnone_len63_notime_val_v1\2021-2022\captured_returns_sw.csv"
# path = "experiment_gml_graph_TEST_split80_gml_cpnone_len63_notime_div_v1"
path = 'exp_lstm_TEST_split80_lstm_cpnone_len20_notime_val_v1'

date = "28-03"

start_year = 2017
end_year = 2022
if STRADDLE:
    # Define file paths for each strategy.
    # path_experiment = rf"C:\Users\Sean\Documents\gml-master\results\{date}\{path}\{start_year}-{end_year}\captured_returns_sw.csv"
    # path_long_only = rf"C:\Users\Sean\Documents\gml-master\results\{date}\long_only\{start_year}-{end_year}\captured_returns_sw.csv"
    # path_tsmom = rf"C:\Users\Sean\Documents\gml-master\results\{date}\tsmom\{start_year}-{end_year}\captured_returns_sw.csv"
    
    path_experiment = rf"C:\Users\Sean\Documents\gml-master\results\29-03\exp_lstm_split80_lstm_cpnone_len20_notime_val_v1\2017-2022\captured_returns_sw.csv"
    path_2 = rf"C:\Users\Sean\Documents\gml-master\results\exp_lstm_gcn_split80_lstm-gcn_cpnone_len20_notime_val_v1\2017-2022\captured_returns_sw.csv"
    path_3 = rf"C:\Users\Sean\Documents\gml-master\results\exp_lstm_gcn_split80_lstm-gcn_benchmark_cpnone_len20_notime_val_v1\2017-2022\captured_returns_sw.csv"

    # Read the CSV files.
    df_lstm = pd.read_csv(path_experiment, parse_dates=["time"])
    df_graph = pd.read_csv(path_2, parse_dates=["time"])
    df_graph_benchmark = pd.read_csv(path_3, parse_dates=["time"])

    def compute_cumulative_returns(df, return_col="captured_returns"):
        """
        Groups the DataFrame by date (time), computes the average return for each day,
        and then calculates the cumulative (geometric) return.
        Cumulative return is computed as: cumulative_return(t) = Π₍ᵢ₌₁₎ᵗ (1 + avg_return(i)) - 1
        """
        # Group by date and average the returns.
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

    # Choose the appropriate return column.
    return_column = "captured_returns" if VOL_SCALING else "returns"

    # Compute cumulative returns for each strategy.
    long_only_returns = compute_cumulative_returns(df_lstm, return_col="returns")
    exp_returns = compute_cumulative_returns(df_lstm, return_col=return_column)
    print(exp_returns.head())
    lstm_gcn_returns = compute_cumulative_returns(df_graph, return_col=return_column)
    tsmom_returns = compute_cumulative_returns(df_graph_benchmark, return_col=return_column)

    # Function to compute volatility-scaled cumulative returns.
    def compute_scaled_cumulative_returns(df_grouped, target_annual_vol=0.15, trading_days=252):
        # Compute the daily volatility (standard deviation of daily average returns)
        daily_vol = df_grouped["avg_captured_returns"].std()
        # Avoid division by zero.
        if daily_vol == 0:
            scale_factor = 1.0
        else:
            scale_factor = target_annual_vol / (daily_vol * np.sqrt(trading_days))
        # Scale the daily returns.
        df_grouped["scaled_daily_returns"] = df_grouped["avg_captured_returns"] * scale_factor
        # Compute the scaled cumulative returns.
        df_grouped["scaled_cumulative_returns"] = (1 + df_grouped["scaled_daily_returns"]).cumprod() - 1
        return df_grouped

    # Compute volatility-scaled cumulative returns for each strategy.
    long_only_returns_scaled = compute_scaled_cumulative_returns(long_only_returns.copy())
    exp_returns_scaled = compute_scaled_cumulative_returns(exp_returns.copy())
    lstm_gcn_returns_scaled = compute_scaled_cumulative_returns(lstm_gcn_returns.copy())
    # (Optionally compute for tsmom_returns as well if desired)

    # Plot the original and volatility-scaled cumulative returns in side-by-side subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Original cumulative returns subplot.
    ax1.plot(long_only_returns["time"], long_only_returns["cumulative_returns"],
             label="Long-Only", lw=1, color="mediumaquamarine")
    ax1.plot(long_only_returns["time"], -long_only_returns["cumulative_returns"],
             label="Short-Only", lw=1, color="blue")
    ax1.plot(exp_returns["time"], exp_returns["cumulative_returns"],
             label="LSTM", lw=1, color="crimson")
    ax1.plot(lstm_gcn_returns["time"], lstm_gcn_returns["cumulative_returns"],
             label="LSTM-GCN", lw=1, color="cyan")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Return")
    ax1.set_title("Raw")
    ax1.legend()
    # (Optionally, add grid or other formatting to ax1)

    # Volatility-scaled cumulative returns subplot.
    ax2.plot(long_only_returns_scaled["time"], long_only_returns_scaled["scaled_cumulative_returns"],
             label="Long-Only", lw=1, color="mediumaquamarine")
    ax2.plot(long_only_returns_scaled["time"], -long_only_returns_scaled["scaled_cumulative_returns"],
             label="Short-Only", lw=1, color="blue")
    ax2.plot(exp_returns_scaled["time"], exp_returns_scaled["scaled_cumulative_returns"],
             label="LSTM", lw=1, color="crimson")
    ax2.plot(lstm_gcn_returns_scaled["time"], lstm_gcn_returns_scaled["scaled_cumulative_returns"],
             label="LSTM-GCN", lw=1, color="cyan")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Return")
    ax2.set_title("Rescaled to 15% Annualized Volatility")
    ax2.legend()
    # (Optionally, add grid or other formatting to ax2)

    plt.tight_layout()
    plt.show()
    
    
    
    # POSITIONS
    
    
    
    #  = df_lstm.groupby("time")["position"].mean().reset_index(name="mean_position")
    lstm_mean_position = df_lstm.groupby("time")["position"].mean().reset_index(name="mean_position")
    graph_mean_position = df_graph.groupby("time")["position"].mean().reset_index(name="mean_position")
    benchmark_mean_position = df_graph_benchmark.groupby("time")["position"].mean().reset_index(name="mean_position")


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ax1.plot(lstm_mean_position["time"], lstm_mean_position["mean_position"],
    #         label="LSTM Mean Position", lw=1)
    ax1.plot(lstm_mean_position["time"],
            lstm_mean_position["mean_position"].rolling(window=5).mean(),
            label="5-day SMA", lw=1, linestyle=":", alpha=1, color="black")
    ax1.plot(lstm_mean_position["time"],
            lstm_mean_position["mean_position"].rolling(window=20).mean(),
            label="20-day SMA", lw=1, linestyle="-", alpha=1, color="dodgerblue")
    ax1.plot(lstm_mean_position["time"],
            lstm_mean_position["mean_position"].rolling(window=60).mean(),
            label="60-day SMA", lw=1, linestyle="-", color="crimson")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Position")
    ax1.set_title("LSTM")
    ax1.legend()
    # ax1.grid(True, which="both", ls="--")

    # ax2.plot(graph_mean_position["time"], graph_mean_position["mean_position"],
    #         label="LSTM-GCN Mean Position", lw=2)
    ax2.plot(graph_mean_position["time"],
            graph_mean_position["mean_position"].rolling(window=5).mean(),
            label="5-day SMA", lw=1, linestyle=":", alpha=0.7, color="black")
    ax2.plot(graph_mean_position["time"],
            graph_mean_position["mean_position"].rolling(window=20).mean(),
            label="20-day SMA", lw=1, linestyle="-", alpha=0.7, color="dodgerblue")
    ax2.plot(graph_mean_position["time"],
            graph_mean_position["mean_position"].rolling(window=60).mean(),
            label="60-day SMA", lw=1, linestyle="-", color="crimson")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Position")
    ax2.set_title("LSTM-GCN")
    ax2.legend()
    # ax2.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()
