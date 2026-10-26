import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

# 1. Load Dataset
house = pd.read_csv("train.csv")
X = house.drop(columns=["Id", "SalePrice"])
y = house["SalePrice"]

# 2. Unified Feature Engineering & Preprocessing
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

# Vectorized imputation without loops
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("None")

# One-hot encoding
X_encoded = pd.get_dummies(X, columns=cat_cols, dtype=int)

# 3. Data Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
# 4. Train Default Model
default_rf = RandomForestRegressor(random_state=42)
default_rf.fit(X_train, y_train)
default_pred = default_rf.predict(X_test)

# 5. Hyperparameter Tuning via Grid Search
param_grid = {
    "n_estimators":[100,200],
    "max_depth": [None, 10, 20],
    "min_samples_split":[2, 5],
    "min_samples_leaf":[1, 2],
    "max_features": [0.7, 1.0],
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# Evaluate the best optimized estimator
best_rf = grid_search.best_estimator_
best_pred = best_rf.predict(X_test)

# 6. Performance Report Evaluation
metrics = {
    "MAE": lambda y, p: mean_absolute_error(y, p),
    "MSE": lambda y, p: mean_squared_error(y, p),
    "RMSE": lambda y, p: np.sqrt(mean_squared_error(y, p)),
    "R2": lambda y, p: r2_score(y, p),
}

results = pd.DataFrame(
    {
        "Metric": metrics.keys(),
        "Default Model": [m(y_test, default_pred) for m in metrics.values()],
        "Optimized Model": [m(y_test, best_pred) for m in metrics.values()],
    }
)

# Format outputs cleanly
print("--- Hyperparameter Search Summary ---")
print(f"Best Configuration: {grid_search.best_params_}")
print(f"Validation RMSE: {np.sqrt(-grid_search.best_score_):.2f}\n")

print("--- Model Performance Comparison ---")
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}" if x < 1 else f"{x:.2f}"))

# 7. Visualization: Evaluation Curve
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_pred, alpha=0.5, color="steelblue", edgecolors="none")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="crimson",
    linestyle="--",
    lw=2,
    label="Perfect Prediction",
)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs. Predicted House Prices (Optimized Model)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()