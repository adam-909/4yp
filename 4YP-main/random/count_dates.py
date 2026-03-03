# import pandas as pd
# import argparse

# def count_rows_by_ticker(csv_file):
#     # Read the CSV file into a DataFrame.
#     df = pd.read_csv(csv_file)
#     # Group by the "ticker" column and count the number of rows per group.
#     counts = df.groupby("ticker").size()
#     return counts

# def main():
#     parser = argparse.ArgumentParser(description="Count rows by ticker from a CSV file.")
#     parser.add_argument("csv_file", help="Path to the CSV file")
#     args = parser.parse_args()

#     counts = count_rows_by_ticker(args.csv_file)
#     print("Number of rows per ticker:")
#     print(counts)

# if __name__ == "__main__":
#     main()
import os
import glob
import pandas as pd

def count_first_date_duplicates(csv_file):
    """
    Reads a CSV file with (at least) two date columns, renames them, and
    counts how many duplicates exist based on the first date column.
    """
    # Read the CSV with no headers; rename columns as needed.
    # Adjust names if your CSV has more/fewer columns.
    df = pd.read_csv(csv_file, header=None, names=["start_date", "val1", "end_date", "val2"])
    
    # Ensure 'start_date' is a string (or convert to datetime if you prefer).
    df["start_date"] = df["start_date"].astype(str)
    
    # Group by the first date column and count how many times each date appears.
    counts = df.groupby("start_date").size()
    # For each date, any rows beyond the first are duplicates => (count - 1).
    num_dup = (counts - 1).clip(lower=0).sum()
    
    return num_dup

def main():
    # Folder containing your CSV files
    directory = "data/new_straddle_data"
    # Find all CSV files in that folder
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    for file_path in csv_files:
        # Count duplicates in each file
        dups = count_first_date_duplicates(file_path)
        print(f"{os.path.basename(file_path)} => {dups} duplicates (based on 'start_date').")

if __name__ == "__main__":
    main()