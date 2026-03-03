import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import yfinance as yf
import tikzplotlib 

TICKER = "AAPL"

START_DATE = "2011-06-01"
END_DATE = "2023-06-01"
VIX_OVERLAY = True  # Set to True to overlay VIX data

basepath = r"C:\Users\Sean\Documents\gml-master"

# Read local CSV for the equity data
df = pd.read_csv(
    rf"{basepath}\data\new_straddle_data\{TICKER}.csv",
    parse_dates=["date"],
)

# Filter the DataFrame between START_DATE and END_DATE
mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
filtered_df = df.loc[mask]

plt.figure(figsize=(6, 6))
ax = plt.gca()

# Plot the equity data and label it for the legend
(line1,) = ax.plot(
    filtered_df["date"],
    filtered_df["straddle_price"],
    color="blue",
    lw=1,
    label=r"$\Delta$-Neutral Straddle Price (\$)",
)
ax.set_xlabel("Year")
ax.set_ylabel(r"$\Delta$-Neutral Straddle Price (\$)")
# Set left axis tick and label colors to blue
ax.tick_params(axis="y", colors="blue")
ax.yaxis.label.set_color("blue")
# Remove the top and right spines
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# Format the x-axis to display one tick per year and auto-rotate date labels
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.gcf().autofmt_xdate()
plt.grid(False)

if VIX_OVERLAY:
    # Download VIX data from yfinance ('^VIX' is the ticker symbol for VIX)
    vix_data = yf.download("^VIX", start=START_DATE, end=END_DATE, progress=False)
    # Create a twin axis for VIX on the right-hand side
    ax2 = ax.twinx()
    # Plot VIX data and label it for the legend
    (line2,) = ax2.plot(
        vix_data.index, vix_data["Close"], color="red", lw=1, label="VIX"
    )
    ax2.set_ylabel("VIX")
    # Set right axis tick and label colors to red
    ax2.tick_params(axis="y", colors="red")
    ax2.yaxis.label.set_color("red")
    # Remove the top spine from the secondary axis
    ax2.spines["top"].set_visible(False)

    # Combine legend entries from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
else:
    ax.legend(loc="upper left")

plt.show()
# tikzplotlib.save("straddle.tex")