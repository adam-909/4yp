import pandas as pd
import matplotlib.pyplot as plt


STRADDLE = True
VOL_SCALING = True

# experiment_gml_TEST_lstm_cpnone_len63_notime_div_v1
# "C:\Users\Sean\Documents\gml-master\results\exp_lstm_TEST_split80_lstm_cpnone_len63_notime_val_v1\2021-2022\captured_returns_sw.csv"
# path = "experiment_gml_graph_TEST_split80_gml_cpnone_len63_notime_div_v1"
path = 'exp_lstm_TEST_split80_lstm_cpnone_len20_notime_val_v1'

date = "28-03"

start_year = 2018
end_year = 2019

if STRADDLE:
    # Define file paths for each strategy.
    path_experiment = rf"C:\Users\Sean\Documents\gml-master\results\{date}\{path}\{start_year}-{end_year}\captured_returns_sw.csv"
    path_long_only = rf"C:\Users\Sean\Documents\gml-master\results\{date}\long_only\{start_year}-{end_year}\captured_returns_sw.csv"
    path_tsmom = rf"C:\Users\Sean\Documents\gml-master\results\{date}\tsmom\{start_year}-{end_year}\captured_returns_sw.csv"

    # Read the CSV files.
    df_experiment = pd.read_csv(path_experiment, parse_dates=["time"])
    df_long_only = pd.read_csv(path_long_only, parse_dates=["time"])
    df_tsmom = pd.read_csv(path_tsmom, parse_dates=["time"])
    


    print(df_experiment.head())

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
    exp_returns = compute_cumulative_returns(df_experiment, return_col=return_column)
    print(exp_returns.head())
    long_only_returns = compute_cumulative_returns(df_long_only, return_col=return_column)
    tsmom_returns = compute_cumulative_returns(df_tsmom, return_col=return_column)

    # Plot the cumulative returns.
    plt.figure(figsize=(10, 6))
    plt.plot(exp_returns["time"], exp_returns["cumulative_returns"],
             label="LSTM", lw=2)
    plt.plot(long_only_returns["time"], long_only_returns["cumulative_returns"],
             label="Long Only", lw=2)
    plt.plot(tsmom_returns["time"], tsmom_returns["cumulative_returns"],
             label="TSMOM", lw=2)

    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Cumulative Returns Comparison")
    plt.yscale("log")  # Use a log scale on the y-axis if desired
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

# else:
#     # Define file paths for each strategy.
#     path_experiment = r"C:\Users\Sean\Documents\gml-master\results\straddles\experiment_gml_lstm_cpnone_len63_notime_div_v1\2021-2023\captured_returns_sw.csv"
#     path_long_only = r"C:\Users\Sean\Documents\gml-master\results\straddles\long_only\2021-2023\captured_returns_sw.csv"
#     path_tsmom = r"C:\Users\Sean\Documents\gml-master\results\straddles\tsmom\2021-2023\captured_returns_sw.csv"

#     # Read the CSV files.
#     df_experiment = pd.read_csv(path_experiment, parse_dates=["time"])
#     df_long_only = pd.read_csv(path_long_only, parse_dates=["time"])
#     df_tsmom = pd.read_csv(path_tsmom, parse_dates=["time"])

#     print(df_experiment.head())

#     def compute_cumulative_returns(df, return_col="captured_returns"):
#         """
#         Groups the DataFrame by date (time), computes the average captured return
#         for each day, and then calculates the cumulative (geometric) return.

#         Cumulative return is computed as:
#           cumulative_return(t) = Π₍ᵢ₌₁₎ᵗ (1 + avg_return(i)) - 1
#         """
#         # Group by date and average the captured returns.
#         df_grouped = (
#             df.groupby("time")[return_col]
#             .mean()
#             .reset_index(name="avg_captured_returns")
#         )
#         # Sort by date.
#         df_grouped = df_grouped.sort_values("time")
#         # Compute the cumulative return.
#         df_grouped["cumulative_returns"] = (
#             1 + df_grouped["avg_captured_returns"]
#         ).cumprod() - 1
#         return df_grouped

#     return_column = "captured_returns" if VOL_SCALING else "returns"

#     # Compute cumulative returns for each strategy.
#     exp_returns = compute_cumulative_returns(df_experiment, return_col=return_column)
#     print(exp_returns.head())
#     long_only_returns = compute_cumulative_returns(df_long_only, return_col=return_column)
#     tsmom_returns = compute_cumulative_returns(df_tsmom, return_col=return_column)

#     # Plot the cumulative returns.
#     plt.figure(figsize=(10, 6))
#     plt.plot(exp_returns["time"], exp_returns["cumulative_returns"],
#              label="LSTM", lw=2)
#     plt.plot(long_only_returns["time"], long_only_returns["cumulative_returns"],
#              label="Long Only", lw=2)
#     plt.plot(tsmom_returns["time"], tsmom_returns["cumulative_returns"],
#              label="TSMOM", lw=2)

#     plt.xlabel("Date")
#     plt.ylabel("Cumulative Return")
#     plt.title("Cumulative Returns Comparison")
#     # plt.yscale("log")  # Use a log scale on the y-axis if desired
#     plt.legend()
#     plt.grid(True, which="both", ls="--")
#     plt.show()
