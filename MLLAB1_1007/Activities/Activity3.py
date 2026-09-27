import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"

df = pd.read_csv(url)

print(df.head())

# Extract Pakistan data
pakistan = df[df["Country/Region"] == "Pakistan"]

print("\nPakistan Data:")
print(pakistan.head())

# Check missing values
print("\nMissing Values:")
print(pakistan.isnull().sum())

# Handle missing values
pakistan = pakistan.dropna()

# Plot confirmed cases over time
plt.plot(pakistan["Date"], pakistan["Confirmed"])

plt.xlabel("Date")
plt.ylabel("Confirmed Cases")
plt.title("COVID-19 Cases in Pakistan")

plt.xticks(rotation=45)

plt.show()

# Find day with highest confirmed cases
highest = pakistan["Confirmed"].max()

print("\nHighest Confirmed Cases:")
print(highest)

print("\nDay with Highest Confirmed Cases:")
print(
    pakistan[pakistan["Confirmed"] == highest]
    [["Date", "Confirmed"]]
)

# Calculate daily cases
pakistan["Daily_Cases"] = pakistan["Confirmed"].diff()

pakistan = pakistan.dropna()

# Boxplot
plt.boxplot(pakistan["Daily_Cases"])

plt.title("Daily COVID-19 Cases")
plt.ylabel("Daily Cases")

plt.show()

# Detect outliers using IQR
Q1 = pakistan["Daily_Cases"].quantile(0.25)
Q3 = pakistan["Daily_Cases"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = pakistan[
    (pakistan["Daily_Cases"] < lower) |
    (pakistan["Daily_Cases"] > upper)
]

print("\nDaily Cases Outliers:")
print(outliers[["Date", "Daily_Cases"]])