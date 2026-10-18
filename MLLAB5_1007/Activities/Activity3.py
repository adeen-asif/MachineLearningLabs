import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Mall_Customers.csv")
print(df.head())

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# K-Means clustering without scaling
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["KMeans_Original"] = kmeans.fit_predict(X)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1],
            c=df["KMeans_Original"])
plt.title("K-Means Without Scaling")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

# DBSCAN clustering
dbscan = DBSCAN(eps=5, min_samples=5)
df["DBSCAN"] = dbscan.fit_predict(X)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1],
            c=df["DBSCAN"])
plt.title("DBSCAN Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means after scaling
kmeans_scaled = KMeans(n_clusters=5, random_state=42, n_init=10)
df["KMeans_Scaled"] = kmeans_scaled.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=df["KMeans_Scaled"])
plt.title("K-Means With Scaling")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.show()

# Elbow Method
inertia = []

for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Add random LuckyNumber feature
np.random.seed(42)
df["LuckyNumber"] = np.random.randint(0, 101, size=len(df))

X_lucky = df[
    ["Annual Income (k$)", "Spending Score (1-100)", "LuckyNumber"]
]

# Apply K-Means with LuckyNumber
kmeans_lucky = KMeans(n_clusters=5, random_state=42, n_init=10)
df["KMeans_Lucky"] = kmeans_lucky.fit_predict(X_lucky)

plt.scatter(df["Annual Income (k$)"],
            df["Spending Score (1-100)"],
            c=df["KMeans_Lucky"])
plt.title("K-Means With LuckyNumber")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

# Display cluster counts
print("Original K-Means Clusters:")
print(df["KMeans_Original"].value_counts().sort_index())

print("\nScaled K-Means Clusters:")
print(df["KMeans_Scaled"].value_counts().sort_index())

print("\nDBSCAN Clusters:")
print(df["DBSCAN"].value_counts().sort_index())

print("\nK-Means With LuckyNumber:")
print(df["KMeans_Lucky"].value_counts().sort_index())

print("\nComparison:")
print("Scaling changes the distance between data points.")
print("Therefore, cluster assignments may change after scaling.")

print("\nLuckyNumber Effect:")
print("LuckyNumber is a random feature with no real meaning.")
print("It can affect distances and change cluster assignments.")