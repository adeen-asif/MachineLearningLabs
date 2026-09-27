import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

# Load Dataset

data = load_breast_cancer()

print("Data Keys:")
print(data.keys())

print("\nTarget Names:")
print(data.target_names)

n_samples, n_features = data.data.shape

print("\nNumber of samples:", n_samples)
print("Number of features:", n_features)

print("\nFeature Names:")
print(data.feature_names)

print("\nInput Shape:")
print(data.data.shape)

print("\nOutput Shape:")
print(data.target.shape)


# Create DataFrame

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

print("\nFirst 5 Rows:")
print(df.head())


# Introduce Missing Values

np.random.seed(42)

random_rows = np.random.choice(
    df.index,
    5,
    replace=False
)

df.loc[random_rows, "mean radius"] = np.nan

print("\nMissing Values:")
print(df["mean radius"].isnull().sum())


# Handle missing values using median
df["mean radius"] = df["mean radius"].fillna(
    df["mean radius"].median()
)

print("\nMissing Values After Handling:")
print(df["mean radius"].isnull().sum())


# Check Class Distribution

print("\nClass Distribution:")
print(df["target"].value_counts())

# 0 = malignant
# 1 = benign


# Handle Imbalanced Data using SMOTE

X = df.drop("target", axis=1)
y = df["target"]

print("\nBefore SMOTE:")
print(y.value_counts())

smote = SMOTE(random_state=42)

X_smote, y_smote = smote.fit_resample(X, y)

print("\nAfter SMOTE:")
print(y_smote.value_counts())


# Detect Outliers using IQR

features = [
    "mean radius",
    "mean texture"
]

for feature in features:

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[feature] < lower) |
        (df[feature] > upper)
    ]

    print("\nOutliers in", feature)
    print(outliers[[feature]])

    # Boxplot
    plt.boxplot(df[feature])

    plt.title("Outliers in " + feature)
    plt.ylabel(feature)

    plt.show()


# Log Transformation

plt.hist(df["mean area"], bins=20)

plt.title("Mean Area Before Log Transformation")
plt.xlabel("Mean Area")
plt.ylabel("Frequency")

plt.show()

# Apply log transformation
df["mean area log"] = np.log1p(
    df["mean area"]
)

plt.hist(df["mean area log"], bins=20)

plt.title("Mean Area After Log Transformation")
plt.xlabel("Log Mean Area")
plt.ylabel("Frequency")

plt.show()


# Feature Scaling

X = df.drop(
    ["target", "mean area log"],
    axis=1
)

y = df["target"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Data:")
print(X_scaled[:5])


# PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Data:")
print(X_pca[:5])

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("PCA")

plt.show()


# Train SVM

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

svm = SVC()

svm.fit(X_train, y_train)

svm_prediction = svm.predict(X_test)

svm_accuracy = accuracy_score(
    y_test,
    svm_prediction
)

print("\nSVM Accuracy:")
print(svm_accuracy)


# Train KNN

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn_prediction = knn.predict(X_test)

knn_accuracy = accuracy_score(
    y_test,
    knn_prediction
)

print("\nKNN Accuracy:")
print(knn_accuracy)


# Compare Accuracy

print("\nSVM Accuracy:", svm_accuracy)
print("KNN Accuracy:", knn_accuracy)


# Conclusion

print("\nConclusion:")
print("Feature scaling is important before using KNN or SVM.")
print("It puts features with different ranges on a similar scale.")