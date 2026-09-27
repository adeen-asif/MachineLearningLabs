import pandas as pd

# Load JSON from local file
df_json = pd.read_json('attendance.json')   # Replace with your file path
print("JSON Data:")
print(df_json.head())

# Load JSON from online source
json_url = 'https://jsonplaceholder.typicode.com/users'
df_online_json = pd.read_json(json_url)
print("\nOnline JSON Data:")
print(df_online_json.head())