import pandas as pd
import matplotlib.pyplot as plt
import os

# Flag to determine which file paths to use.
STRADDLE = True

# Set the folder (experiment) name.
path = 'exp_lstm_TEST_split80_lstm_cpnone_len63_notime_val_v1'

if STRADDLE:
    # Define file paths for each strategy.
    path_experiment = rf"C:\Users\Sean\Documents\gml-master\results\{path}\2021-2022\captured_returns_fw.csv"
    path_long_only = rf"C:\Users\Sean\Documents\gml-master\results\long_only\2021-2022\captured_returns_sw.csv"
    path_tsmom = rf"C:\Users\Sean\Documents\gml-master\results\tsmom\2021-2022\captured_returns_sw.csv"

    # Read the CSV files.
    df_experiment = pd.read_csv(path_experiment, parse_dates=["time"])
    df_long_only = pd.read_csv(path_long_only, parse_dates=["time"])
    df_tsmom = pd.read_csv(path_tsmom, parse_dates=["time"])

    print(df_experiment.head())

    def compute_cumulative_returns(df, return_col="captured_returns"):
        """
        Groups the DataFrame by date ('time'), computes the average captured return
        for each day, and then calculates the cumulative (geometric) return:
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
        df_grouped["cumulative_returns"] = (1 + df_grouped["avg_captured_returns"]).cumprod() - 1
        return df_grouped

    # Specify the name of the identifier column (change as needed; e.g., "ticker")
    id_col = "identifier"

    # Compute the union of identifiers across the three dataframes.
    ids_exp = set(df_experiment[id_col].unique()) if id_col in df_experiment.columns else set()
    ids_long = set(df_long_only[id_col].unique()) if id_col in df_long_only.columns else set()
    ids_tsmom = set(df_tsmom[id_col].unique()) if id_col in df_tsmom.columns else set()
    all_ids = ids_exp | ids_long | ids_tsmom

    # For each identifier, compute and plot cumulative returns for each strategy.
    for identifier in all_ids:
        plt.figure(figsize=(10, 6))
        
        # Filter each dataframe by the current identifier.
        df_exp_id = df_experiment[df_experiment[id_col] == identifier]
        df_long_id = df_long_only[df_long_only[id_col] == identifier]
        df_tsmom_id = df_tsmom[df_tsmom[id_col] == identifier]
        
        # Plot cumulative returns if there is data for that identifier.
        if not df_exp_id.empty:
            exp_returns = compute_cumulative_returns(df_exp_id)
            plt.plot(exp_returns["time"], exp_returns["cumulative_returns"],
                     label="Experiment Strategy", lw=2)
        if not df_long_id.empty:
            long_returns = compute_cumulative_returns(df_long_id)
            plt.plot(long_returns["time"], long_returns["cumulative_returns"],
                     label="Long Only Strategy", lw=2)
        if not df_tsmom_id.empty:
            tsmom_returns = compute_cumulative_returns(df_tsmom_id)
            plt.plot(tsmom_returns["time"], tsmom_returns["cumulative_returns"],
                     label="TSMOM Strategy", lw=2)
        
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.title(f"Cumulative Returns Comparison for {identifier}")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()

else:
    # Define file paths for each strategy (non-straddle case).
    path_experiment = r"C:\Users\Sean\Documents\gml-master\results\straddles\experiment_gml_lstm_cpnone_len63_notime_div_v1\2021-2023\captured_returns_sw.csv"
    path_long_only = r"C:\Users\Sean\Documents\gml-master\results\straddles\long_only\2021-2023\captured_returns_sw.csv"
    path_tsmom = r"C:\Users\Sean\Documents\gml-master\results\straddles\tsmom\2021-2023\captured_returns_sw.csv"

    # Read the CSV files.
    df_experiment = pd.read_csv(path_experiment, parse_dates=["time"])
    df_long_only = pd.read_csv(path_long_only, parse_dates=["time"])
    df_tsmom = pd.read_csv(path_tsmom, parse_dates=["time"])

    print(df_experiment.head())

    def compute_cumulative_returns(df, return_col="captured_returns"):
        """
        Groups the DataFrame by date ('time'), computes the average captured return
        for each day, and then calculates the cumulative (geometric) return.
        """
        df_grouped = (
            df.groupby("time")[return_col]
            .mean()
            .reset_index(name="avg_captured_returns")
        )
        df_grouped = df_grouped.sort_values("time")
        df_grouped["cumulative_returns"] = (1 + df_grouped["avg_captured_returns"]).cumprod() - 1
        return df_grouped

    id_col = "identifier"  # Change if needed.
    ids_exp = set(df_experiment[id_col].unique()) if id_col in df_experiment.columns else set()
    ids_long = set(df_long_only[id_col].unique()) if id_col in df_long_only.columns else set()
    ids_tsmom = set(df_tsmom[id_col].unique()) if id_col in df_tsmom.columns else set()
    all_ids = ids_exp | ids_long | ids_tsmom

    for identifier in all_ids:
        plt.figure(figsize=(10, 6))
        
        df_exp_id = df_experiment[df_experiment[id_col] == identifier]
        df_long_id = df_long_only[df_long_only[id_col] == identifier]
        df_tsmom_id = df_tsmom[df_tsmom[id_col] == identifier]
        
        if not df_exp_id.empty:
            exp_returns = compute_cumulative_returns(df_exp_id)
            plt.plot(exp_returns["time"], exp_returns["cumulative_returns"],
                     label="Experiment Strategy", lw=2)
        if not df_long_id.empty:
            long_returns = compute_cumulative_returns(df_long_id)
            plt.plot(long_returns["time"], long_returns["cumulative_returns"],
                     label="Long Only Strategy", lw=2)
        if not df_tsmom_id.empty:
            tsmom_returns = compute_cumulative_returns(df_tsmom_id)
            plt.plot(tsmom_returns["time"], tsmom_returns["cumulative_returns"],
                     label="TSMOM Strategy", lw=2)
        
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.title(f"Cumulative Returns Comparison for {identifier}")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()
