# Robust monkey-patch for webcolors.CSS3_HEX_TO_NAMES
import webcolors
if not hasattr(webcolors, "CSS3_HEX_TO_NAMES"):
    try:
        if hasattr(webcolors, "css3_hex_to_names"):
            webcolors.CSS3_HEX_TO_NAMES = webcolors.css3_hex_to_names
        elif hasattr(webcolors, "css3_names_to_hex"):
            webcolors.CSS3_HEX_TO_NAMES = {v: k for k, v in webcolors.css3_names_to_hex.items()}
        else:
            # Fallback: use matplotlib's CSS4_COLORS as an approximation.
            import matplotlib.colors as mcolors
            webcolors.CSS3_HEX_TO_NAMES = {v: k for k, v in mcolors.CSS4_COLORS.items()}
    except Exception as e:
        raise AttributeError("Cannot patch webcolors.CSS3_HEX_TO_NAMES: " + str(e))

import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib
from matplotlib.dates import MonthLocator, DateFormatter
import sys

# Default CSV file path
data_path = r"C:\Users\Sean\Documents\gml-master\data\straddle_features\features.csv"

# Dictionary mapping tickers to Bloomberg sectors.
from settings.default import BBG_SECTORS

def compute_statistics(series):
    # Drop missing values
    data = series.dropna()
    return {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
    }

def main(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if the expected columns exist
    for col in ['daily_returns', 'norm_monthly_return', 'norm_annual_return', 'ticker', 'date']:
        if col not in df.columns:
            print(f"Error: '{col}' column not found in CSV file.")
            return
    
    # Compute statistics (optional)
    daily_returns_stats = compute_statistics(df['daily_returns'])
    weekly_returns_stats = compute_statistics(df['norm_monthly_return'])
    annual_returns_stats = compute_statistics(df['norm_annual_return'])
    
    # Map tickers to their Bloomberg sectors using the provided dictionary.
    df['sector'] = df['ticker'].map(BBG_SECTORS)
    
    # Convert 'date' to datetime and sort by date.
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    # Filter the DataFrame for the year 2015.
    df_2015 = df[(df['date'].dt.year == 2015)]
    
    if df_2015.empty:
        print("No data available for the year 2015.")
        return
    
    # Group by date and sector, then compute the mean norm_annual_return for each day.
    sector_daily_avg = df_2015.groupby(['date', 'sector'])['norm_annual_return'].mean().unstack('sector')
    
    data_to_plot = sector_daily_avg

    plt.figure(figsize=(10, 6))
    for sector in data_to_plot.columns:
        plt.plot(data_to_plot.index, data_to_plot[sector], label=sector, linewidth=1)
    
    # Set x-axis limits from January 1, 2015 to January 1, 2016.
    plt.xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2016-01-01"))
    
    ax = plt.gca()
    # Set major ticks every 2 months with labels in "YYYY-MM" format.
    ax.xaxis.set_major_locator(MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    # Set minor ticks every month.
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.spines[['right', 'top']].set_visible(False)
    
    # Add axis labels and legend.
    plt.xlabel("Date")
    plt.ylabel("Normalized Monthly Return")
    plt.legend(title="Sector", prop={'size': 8}, loc='upper left')
    plt.tight_layout()
    
    # Export the plot to a TikZ file.
    tikzplotlib.save("my_plot.tex")
    
    plt.show()

if __name__ == "__main__":
    # Use command-line argument if provided, otherwise use the default data_path.
    csv_file = sys.argv[1] if len(sys.argv) > 1 else data_path
    main(csv_file)
