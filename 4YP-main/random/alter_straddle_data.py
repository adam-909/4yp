import numpy as np
import pandas as pd
import os 

input_dir = r"data/straddle_data/old"
output_dir = r"data/straddle_data"

import os
import pandas as pd

# Map of new tickers to their old ticker symbols
OLD_TICKER_MAP = {
    "BKNG": "PCLN",  # Booking Holdings (formerly Priceline)
    "AVGO": "BRCM",  # Broadcom (formerly traded as BRCM)
    "LIN":  "PX",    # Linde (merged with Praxair PX)
    "MDLZ": "KFT",   # Mondelēz (spun off from Kraft Foods KFT)
    "NEE":  "FPL",   # NextEra Energy (formerly FPL Group)
    "RTX":  "UTX",   # Raytheon Technologies (formerly United Technologies UTX)
    "TMUS": "PCS",   # T-Mobile US (via MetroPCS PCS reverse merger)
    "GOOGL": "GOOG", 
    "META": "FB",
}

os.makedirs(output_dir, exist_ok=True)

def filter_call_put_tickers(input_dir, output_dir, OLD_TICKER_MAP):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path)

            # 1) Extract the portion of call_symbol before the first space
            df["call_ticker"] = df["call_symbol"].str.split(" ").str[0]

            # 2) Extract the portion of put_symbol before the first space
            df["put_ticker"] = df["put_symbol"].str.split(" ").str[0]

            # Build a filter for the call side
            call_mask = (
                (df["ticker"] == df["call_ticker"]) |
                (
                    df["ticker"].isin(OLD_TICKER_MAP.keys()) &
                    (df["call_ticker"] == df["ticker"].map(OLD_TICKER_MAP))
                )
            )

            # Build a filter for the put side
            put_mask = (
                (df["ticker"] == df["put_ticker"]) |
                (
                    df["ticker"].isin(OLD_TICKER_MAP.keys()) &
                    (df["put_ticker"] == df["ticker"].map(OLD_TICKER_MAP))
                )
            )

            # Combine the two with OR, so any row matching call OR put is included
            combined_mask = call_mask & put_mask

            filtered_df = df[combined_mask].copy()

            # Drop the helper columns
            filtered_df.drop(columns=["call_ticker", "put_ticker"], inplace=True)

            # Save filtered data
            output_path = os.path.join(output_dir, file_name)
            filtered_df.to_csv(output_path, index=False)

            print(f"Processed and saved: {file_name}")
            
filter_call_put_tickers(input_dir=input_dir, output_dir=output_dir, OLD_TICKER_MAP=OLD_TICKER_MAP)
