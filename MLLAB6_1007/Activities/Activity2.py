import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from ucimlrepo import fetch_ucirepo

# 1. Load Dataset
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine into a single dataframe for unified cleaning
heart = X.assign(target=(y.iloc[:, 0] > 0).astype(int))

# 2. Data Cleaning & Encoding
# Impute missing values with the column median
heart = heart.fillna(heart.median())

# One-hot encode explicit categorical features
categorical_cols = ["cp", "restecg", "slope", "thal"]
heart = pd.get_dummies(
    heart, columns=categorical_cols, drop_first=True, dtype=int
)

# 3. Data Split
X_clean = heart.drop(columns=["target"])
y_clean = heart["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, stratify=y_clean, random_state=42
)

# 4. Hyperparameter Optimization via Grid Search
param_grid = {
    "n_estimators":[100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split":[2, 5],
    "min_samples_leaf":[1,2],
    "max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# Evaluate the best optimized estimator
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# 5. Performance Report
print("--- Grid Search Summary ---")
print(f"Best Configuration: {grid_search.best_params_}")
print(f"Validation Accuracy: {grid_search.best_score_:.4f}\n")

print("--- Test Set Performance ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}\n")

# 6. Evaluation Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_title("Model Confusion Matrix")
ax1.set_xlabel("Predicted Label")
ax1.set_ylabel("True Label")

# Plot 2: Top Feature Importances
importances = pd.Series(
    best_rf.feature_importances_, index=X_clean.columns
).sort_values()
importances.tail(10).plot(kind="barh", ax=ax2, color="steelblue")
ax2.set_title("Top 10 Feature Importances")
ax2.set_xlabel("Relative Importance Score")

plt.tight_layout()
plt.show()

print("--- Top 3 Influential Predictors ---")
print(importances.sort_values(ascending=False).head(3).to_string())