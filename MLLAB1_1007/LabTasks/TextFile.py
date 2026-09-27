# Loading from Text File
import pandas as pd

# Load text file with custom delimiter
df_text = pd.read_csv('data.txt', delimiter='\t')   # For tab-separated
print("Text File Data:")
print(df_text.head())