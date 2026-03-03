import os
import pandas as pd

output_dir = "data/straddle_data"
# Define the expected date as a Timestamp for robust comparison
expected_date = pd.Timestamp("2010-06-01")

files_with_wrong_date = []

# Iterate over CSV files in the output directory
for file_name in os.listdir(output_dir):
    if file_name.lower().endswith(".csv"):
        file_path = os.path.join(output_dir, file_name)
        df = pd.read_csv(file_path)
        
        # Skip files that are empty or missing the "date" column
        if df.empty or "date" not in df.columns:
            continue
        
        # Parse the date from the first row using pd.to_datetime for consistency
        try:
            first_date = pd.to_datetime(df.iloc[0]["date"])
        except Exception as e:
            print(f"Error parsing date in {file_name}: {e}")
            files_with_wrong_date.append(file_name)
            continue
        
        # If the date doesn't match the expected date, add the file name to the list
        if first_date != expected_date:
            files_with_wrong_date.append(file_name)

print("Files with first row date not equal to June 1, 2010:")
for f in files_with_wrong_date:
    print(f)
