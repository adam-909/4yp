import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import yfinance as yf
import datetime

# Define your start and end dates
START_DATE = "2000-01-01"
END_DATE = "2023-01-01"



# Download VIX data from yfinance
vix_data = yf.download("^VIX", start=START_DATE, end=END_DATE)

# Rename the "Close" column to "price" for consistency
vix_data.rename(columns={"Close": "price"}, inplace=True)

# Filter the DataFrame between START_DATE and END_DATE (if necessary)
mask = (vix_data.index >= START_DATE) & (vix_data.index <= END_DATE)
filtered_df = vix_data.loc[mask]

# Plot the VIX price (Close) against the date index
plt.figure(figsize=(10, 6))
plt.plot(filtered_df.index, filtered_df["price"], lw=2)
plt.xlabel("Year")
plt.ylabel("VIX Level")

# Format the x-axis to display one tick per year
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Hide top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Rotate date labels for better readability
plt.gcf().autofmt_xdate()

# Enable only horizontal grid lines (y-axis grid)
plt.grid(True, axis='y')

plt.show()