import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

TICKER = "XOM"
START_DATE = "2010-01-01"
END_DATE = "2020-01-01"


basepath = r"C:\Users\Sean\Documents\gml-master"

df = pd.read_csv(
    rf"{basepath}\data\new_equity_data\{TICKER}.csv",
     parse_dates=["date"],
    #  index_col="date"
)

mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
filtered_df = df.loc[mask]

# Plot the price against date
plt.figure(figsize=(10, 6))
plt.plot(filtered_df["date"], filtered_df["price"])
                                        #   , lw=2)
plt.xlabel("Year")
plt.ylabel("Equity Price ($)")
# plt.title(f"Price vs Date from {START_DATE} to {END_DATE}")

# Format x-axis to display one tick per year
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.gcf().autofmt_xdate()  # Auto rotates the date labels for better display
plt.grid(False)
plt.show()