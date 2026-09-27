import pandas as pd

# CSV 
# Load from local file
df_csv = pd.read_csv('cars.csv') # Replace with your file path
print("CSV Data:")
print(df_csv.head())

# Load from online source
online_csv_url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv'
df_online_csv = pd.read_csv(online_csv_url)
print("\nOnline CSV Data:")
print(df_online_csv.head())