import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("creditcard.csv")
print(df.head())

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train KNN with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# Tune k using GridSearchCV
grid = {
    "n_neighbors": [3, 5, 7]
}

knn_cv = GridSearchCV(
    KNeighborsClassifier(), grid, cv=3, scoring="f1", n_jobs=-1
)
knn_cv.fit(X_train, y_train)

print("Best Parameters:", knn_cv.best_params_)

# Evaluate the best model
y_pred = knn_cv.predict(X_test)

print("Best Model Accuracy:", accuracy_score(y_test, y_pred))
print("Best Model F1-Score:", f1_score(y_test, y_pred))
