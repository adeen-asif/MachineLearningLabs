import pandas as pd
data = {
'Name': ['Ali', 'Ahmad', 'Raha', 'Mirha'],
'Age': [20, 25, 30, 35],
'City': ['Lahore', 'Faisalabad', 'Islamabad', 'Karachi'],
'Salary': [50000, 55000, 40000, 60000]
}

df_manual = pd.DataFrame(data)
print("Manual DataFrame:")
print(df_manual)