import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import sys

# Default CSV file path
data_path = r"C:\Users\Sean\Documents\gml-master\data\straddle_features\features.csv"

def compute_statistics(series):
    # Drop missing values
    data = series.dropna()
    return {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'skew': skew(data),
        'kurtosis': kurtosis(data)
    }

def print_latex_table(bin_edges, densities, histogram_name):
    print(f"\\begin{{table}}[h]")
    print(f"\\centering")
    print(f"\\caption{{Histogram Data for {histogram_name}}}")
    print(f"\\begin{{tabular}}{{cccc}}")
    print("\\hline")
    print("Bin Lower & Bin Upper & Bin Width & Density \\\\")
    print("\\hline")
    for i in range(len(densities)):
        lower_edge = bin_edges[i]
        upper_edge = bin_edges[i+1]
        width = upper_edge - lower_edge
        dens = densities[i]
        print(f"{lower_edge:.4f} & {upper_edge:.4f} & {width:.4f} & {dens:.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print(f"\\end{{table}}")
    print("\n")  # Add spacing between tables

def main(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if the expected columns exist
    required_columns = ['daily_returns', 'norm_monthly_return', 'norm_annual_return']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: '{col}' column not found in CSV file.")
            return
    
    # Compute statistics
    daily_returns_stats = compute_statistics(df['daily_returns'])
    weekly_returns_stats = compute_statistics(df['norm_monthly_return'])
    monthly_returns_stats = compute_statistics(df['norm_annual_return'])
    
    # Print summary statistics (optional)
    print("Daily Returns Statistics:")
    for key, value in daily_returns_stats.items():
        print(f"  {key}: {value}")
        
    print("\nWeekly Returns Statistics:")
    for key, value in weekly_returns_stats.items():
        print(f"  {key}: {value}")
    
    print("\nMonthly Returns Statistics:")
    for key, value in monthly_returns_stats.items():
        print(f"  {key}: {value}")
    
    # Create a figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Define function to compute x-axis limits based on quantiles (99.9% of the data)
    def get_xlim(series, lower_quantile=0.0005, upper_quantile=0.9995):
        data = series.dropna()
        return data.quantile(lower_quantile), data.quantile(upper_quantile)
    
    n_bins = 50

    # --- Daily Returns ---
    daily_data = df['daily_returns']
    lower_daily, upper_daily = get_xlim(daily_data)
    axes[0].hist(daily_data.dropna(), bins=n_bins, edgecolor='black', density=True)
    axes[0].set_title("Histogram of Daily Returns")
    axes[0].set_xlabel("Daily Returns")
    axes[0].set_ylabel("Probability Density")
    axes[0].set_xlim(lower_daily, upper_daily)
    axes[0].grid(True)
    
    # Compute histogram for daily returns (using same bin range as in the plot)
    daily_hist, daily_edges = np.histogram(daily_data.dropna(), bins=n_bins, 
                                           range=(lower_daily, upper_daily), density=True)
    
    # --- Weekly Returns ---
    weekly_data = df['norm_monthly_return']
    lower_weekly, upper_weekly = get_xlim(weekly_data)
    axes[1].hist(weekly_data.dropna(), bins=n_bins, edgecolor='black', density=True)
    axes[1].set_title("Histogram of Weekly Returns")
    axes[1].set_xlabel("Weekly Returns (Normalized Monthly Return)")
    axes[1].set_ylabel("Probability Density")
    axes[1].set_xlim(lower_weekly, upper_weekly)
    axes[1].grid(True)
    
    # Compute histogram for weekly returns
    weekly_hist, weekly_edges = np.histogram(weekly_data.dropna(), bins=n_bins,
                                             range=(lower_weekly, upper_weekly), density=True)
    
    # --- Monthly Returns ---
    monthly_data = df['norm_annual_return']
    lower_monthly, upper_monthly = get_xlim(monthly_data)
    axes[2].hist(monthly_data.dropna(), bins=n_bins, edgecolor='black', density=True)
    axes[2].set_title("Histogram of Monthly Returns")
    axes[2].set_xlabel("Monthly Returns (Normalized Annual Return)")
    axes[2].set_ylabel("Probability Density")
    axes[2].set_xlim(lower_monthly, upper_monthly)
    axes[2].grid(True)
    
    # Compute histogram for monthly returns
    monthly_hist, monthly_edges = np.histogram(monthly_data.dropna(), bins=n_bins,
                                               range=(lower_monthly, upper_monthly), density=True)
    
    # Adjust layout and display plot
    plt.tight_layout()
    plt.show()
    
    # Print LaTeX tables with bin information for each histogram
    print("%% LaTeX table for Daily Returns Histogram")
    print_latex_table(daily_edges, daily_hist, "Daily Returns")
    
    print("%% LaTeX table for Weekly Returns Histogram")
    print_latex_table(weekly_edges, weekly_hist, "Weekly Returns")
    
    print("%% LaTeX table for Monthly Returns Histogram")
    print_latex_table(monthly_edges, monthly_hist, "Monthly Returns")
    
    # Process dates and ticker statistics if needed
    df['date'] = pd.to_datetime(df['date'])
    ticker_stats = df.groupby('ticker').agg(
        num_rows=('ticker', 'size'),
        start_date=('date', 'min'),
        end_date=('date', 'max')
    )
    print(ticker_stats)

if __name__ == "__main__":
    # Use command-line argument if provided, otherwise use the default data_path.
    csv_file = sys.argv[1] if len(sys.argv) > 1 else data_path
    main(csv_file)
