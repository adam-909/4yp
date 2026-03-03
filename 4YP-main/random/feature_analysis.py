# Date,straddle_price,exdate,moneyness,daily_returns,daily_vol,target_returns,norm_daily_return,norm_monthly_return,norm_quarterly_return,norm_biannual_return,norm_annual_return,macd_2_8,macd_4_16,macd_8_32,day_of_week,day_of_month,month_of_year,year,log_moneyness,time_to_expiry,ticker,date

import pandas as pd

# Assume your data is loaded into a DataFrame called df
# For example: df = pd.read_csv("your_data.csv")

# Define the aggregation functions for the two columns
agg_funcs = {
    'daily_returns': ['mean', 'std', 'skew'],
    'target_returns': ['mean', 'std', 'skew']
}

df = pd.read_csv(r"C:\Users\Sean\Documents\gml-master\data\straddle_features\features.csv")

# Group by ticker and calculate the required statistics
result = df.groupby('ticker').agg(agg_funcs)

# Print the results
print(result)

# result.to_csv("random/output.csv")

df_2018 = df[df['year'] == 2018]

# Define the aggregation functions for the two columns
agg_funcs = {
    'daily_returns': ['mean', 'std', 'skew'],
    'target_returns': ['mean', 'std', 'skew']
}

# Group by ticker and calculate the required statistics for 2018
result_2018 = df_2018.groupby('ticker').agg(agg_funcs)

# Ensure the "random" directory exists

# Save the result to a CSV file in the "random/" directory
result_2018.to_csv("random/output_2018.csv")
