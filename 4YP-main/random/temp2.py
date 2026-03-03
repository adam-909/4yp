import pandas as pd
import matplotlib.pyplot as plt

# 1. Read the CSV (make sure to point to the correct file path).
#    We parse_dates on the 'time' column to have proper DateTime objects.
path = r"C:\Users\Sean\Documents\gml-master\results\experiment_gml_TEST_lstm_cpnone_len63_notime_div_v1\2021-2022\captured_returns_sw.csv"
df = pd.read_csv(path, parse_dates=["time"])

# 2. For each stock (identifier), compute mean, std, skew of "captured_returns"
stats_df = df.groupby("identifier")["captured_returns"].agg(["mean", "std", "skew"])
print(stats_df[:50])
print(stats_df[51:])

# # 3. (Optional) If you still want the averaged cumulative returns across all stocks:
# df_grouped = df.groupby("time")["captured_returns"].mean().reset_index(name="avg_captured_returns")
# df_grouped = df_grouped.sort_values("time")
# df_grouped["cumulative_returns"] = (1 + df_grouped["avg_captured_returns"]).cumprod() - 1

# plt.figure(figsize=(10,6))
# plt.plot(df_grouped["time"], df_grouped["cumulative_returns"], label="Avg Strategy Cumulative Return")
# plt.xlabel("Date")
# plt.ylabel("Cumulative Return")
# plt.title("Averaged TSMOM Strategy Cumulative Returns")
# plt.legend()
# plt.grid(True)
# plt.show()
