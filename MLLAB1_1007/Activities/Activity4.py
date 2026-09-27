import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
students = pd.read_csv("students.csv")

# Load JSON
attendance = pd.read_json("attendance.json")

# Load Excel
extra = pd.read_excel("extra.xlsx")

print("Students:")
print(students)

print("\nAttendance:")
print(attendance)

print("\nExtra:")
print(extra)

# Merge all DataFrames
df = students.merge(attendance, on="Name")

df = df.merge(extra, on="Name")

print("\nMerged Data:")
print(df)

# Scatter Plot
plt.scatter(
    df["Marks"],
    df["Attendance"]
)

# Highlight attendance below 70%
low_attendance = df[df["Attendance"] < 70]

plt.scatter(
    low_attendance["Marks"],
    low_attendance["Attendance"]
)

# Label students
for i in range(len(low_attendance)):
    plt.text(
        low_attendance["Marks"].iloc[i],
        low_attendance["Attendance"].iloc[i],
        low_attendance["Name"].iloc[i]
    )

plt.xlabel("Marks")
plt.ylabel("Attendance")
plt.title("Marks vs Attendance")

plt.show()

# One-Hot Encoding on Name
encoded = pd.get_dummies(df["Name"])

print("\nOne-Hot Encoding:")
print(encoded)