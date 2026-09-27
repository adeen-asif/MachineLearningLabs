import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Create dataset
data = {
    "Province": ["Punjab", "Sindh", "KPK", "Balochistan"],
    "Population": [127.7, 57.9, 40.9, None],
    "Literacy": [64, None, 55, 46],
    "Region": ["East", "South", "North", "West"]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Handle missing values
# Drop one row
df = df.dropna(subset=["Population"])

# Fill missing literacy rate with median
df["Literacy"] = df["Literacy"].fillna(df["Literacy"].median())

print("\nAfter Handling Missing Values:")
print(df)

# Label Encoding
le = LabelEncoder()
df["Region_Label"] = le.fit_transform(df["Region"])

print("\nLabel Encoding:")
print(df)

# One-Hot Encoding
one_hot = pd.get_dummies(df["Region"])

print("\nOne-Hot Encoding:")
print(one_hot)

# Scatter Plot
plt.scatter(df["Population"], df["Literacy"])

for i in range(len(df)):
    plt.text(
        df["Population"].iloc[i],
        df["Literacy"].iloc[i],
        df["Province"].iloc[i]
    )

plt.xlabel("Population (millions)")
plt.ylabel("Literacy Rate (%)")
plt.title("Population vs Literacy Rate")
plt.show()

# Detect Outlier using IQR in literacy rate
Q1 = df["Literacy"].quantile(0.25)
Q3 = df["Literacy"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["Literacy"] < lower) |
    (df["Literacy"] > upper)
]

print("\nLiteracy Rate Outliers:")
print(outliers)