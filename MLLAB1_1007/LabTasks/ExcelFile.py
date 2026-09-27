import pandas as pd

# Load Excel file
df_excel = pd.read_excel('extra.xlsx')   
print("Excel Data:")
print(df_excel.head())
