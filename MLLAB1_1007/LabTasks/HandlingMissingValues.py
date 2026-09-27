import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create a sample dataset with missing values
data = {
    'A': [1, 2, None, 4, 5],
    'B': [6, None, 8, 9, 10],
    'C': [11, 12, 13, None, 15]
}
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Handling Missing Values

# Method 1: Drop rows with missing values
df_dropna_rows = df.dropna(axis=0)

# Method 2: Drop columns with missing values
df_dropna_columns = df.dropna(axis=1)

# Method 3: Fill missing values with the median
df_fill_median = df.fillna(df.median())

# Display results of missing value handling
print("\nDataFrame after dropping rows with missing values:")
print(df_dropna_rows)

print("\nDataFrame after dropping columns with missing values:")
print(df_dropna_columns)

print("\nDataFrame after filling missing values with the median:")
print(df_fill_median)
