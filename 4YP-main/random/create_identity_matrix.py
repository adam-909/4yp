import pandas as pd
import numpy as np
import os

# Define the list of ticker labels
tickers = [
    "AAPL", "ABT", "ACN", "ADBE", "AIG", "AMGN", "AMT", "AMZN", "AVGO", "AXP",
    "BAC", "BA", "BKNG", "BK", "BLK", "BMY", "BRKB", "CAT", "CL", "CMCSA",
    "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", "C", "DE", "DHR",
    "DIS", "DUK", "EMR", "FDX", "F", "GD", "GE", "GILD", "GOOGL", "GS",
    "HD", "HON", "IBM", "INTC", "INTU", "JNJ", "JPM", "KO", "LIN", "LLY",
    "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK",
    "MSFT", "MS", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG",
    "PM", "QCOM", "RTX", "SBUX", "SO", "SPG", "TGT", "TMO", "TMUS", "TXN",
    "T", "UNH", "UNP", "UPS", "USB", "VZ", "V", "WFC", "WMT", "XOM"
]

# Create an identity matrix with the same dimension as the number of tickers
identity_matrix = np.eye(len(tickers), dtype=int)

# Create a DataFrame using the identity matrix with tickers as both row and column labels
df_identity = pd.DataFrame(identity_matrix, index=tickers, columns=tickers)

# Ensure the output directory exists
os.makedirs("data/graph_structure", exist_ok=True)

# Save the DataFrame as a CSV file
df_identity.to_csv("data/graph_structure/identity.csv")