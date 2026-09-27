import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)

data = {
    "ID": range(1, 101),
    "Math": np.random.randint(40, 101, 100),
    "Science": np.random.randint(40, 101, 100),
    "English": np.random.randint(40, 101, 100),
    "Grade": np.random.choice(["A", "B", "C"], 100)
}

df = pd.DataFrame(data)

# Add some missing marks
df.loc[5, "Math"] = np.nan
df.loc[20, "Science"] = np.nan
df.loc[50, "English"] = np.nan

print("Original Data:")
print(df.head())

# Fill missing marks with mean
df["Math"] = df["Math"].fillna(df["Math"].mean())
df["Science"] = df["Science"].fillna(df["Science"].mean())
df["English"] = df["English"].fillna(df["English"].mean())

# Encode Grade
mapping = {
    "A": 3,
    "B": 2,
    "C": 1
}

df["Grade_Encoded"] = df["Grade"].map(mapping)

print("\nAfter Handling Missing Values:")
print(df.head())

# Histogram of Math scores
plt.hist(df["Math"], bins=10)
plt.xlabel("Math Scores")
plt.ylabel("Number of Students")
plt.title("Math Scores")
plt.show()

# Calculate total score
df["Total"] = (
    df["Math"] +
    df["Science"] +
    df["English"]
)

# Boxplot
plt.boxplot(df["Total"])
plt.ylabel("Total Score")
plt.title("Total Score Boxplot")
plt.show()

# Detect unusually high/low total scores
Q1 = df["Total"].quantile(0.25)
Q3 = df["Total"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["Total"] < lower) |
    (df["Total"] > upper)
]

print("\nUnusual Students:")
print(outliers[["ID", "Total"]])

# Find which grade performed best
print("\nAverage Total Score by Grade:")
print(df.groupby("Grade")["Total"].mean())