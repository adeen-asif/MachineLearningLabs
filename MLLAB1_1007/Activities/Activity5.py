import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Given dataset
data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Handle missing Values

df["Age"] = df["Age"].fillna(df["Age"].median())


df["Salary"] = df["Salary"].fillna(df["Salary"].median())

print("\nAfter Handling Missing Values:")
print(df)

# Encode Department
le = LabelEncoder()

df["Department_Encoded"] = le.fit_transform(
    df["Department"]
)

print("\nAfter Encoding Department:")
print(df)

# Salary Boxplot
plt.boxplot(df["Salary"])

plt.title("Salary Boxplot")
plt.ylabel("Salary")

plt.show()

# Detect Salary Outliers
Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["Salary"] < lower) |
    (df["Salary"] > upper)
]

print("\nSalary Outliers:")
print(outliers[["Name", "Salary"]])

# Check Zara's Salary
zara = df[df["Name"] == "Zara"]

if zara["Salary"].iloc[0] > upper:
    print("\nZara's salary is an outlier.")
else:
    print("\nZara's salary is not an outlier.")