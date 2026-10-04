import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
data = pd.read_csv("headbrain.csv")
data.head()

# Select features and target
X = data[["Gender", "Age Range", "Head Size(cm^3)"]].values
y = data["Brain Weight(grams)"].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)
print("Linear Regression RMSE:", linear_rmse)
print("Linear Regression R2 Score:", linear_r2)

# Polynomial Regression
degrees = [2, 3, 4]
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    poly_pred = poly_model.predict(X_test_poly)
    rmse = np.sqrt(mean_squared_error(y_test, poly_pred))
    r2 = r2_score(y_test, poly_pred)
    print("Degree", degree)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

# Plot fitted curves using Head Size as the independent variable
X_plot = data[["Head Size(cm^3)"]].values
y_plot = data["Brain Weight(grams)"].values
sort_index = X_plot[:, 0].argsort()
X_plot = X_plot[sort_index]
y_plot = y_plot[sort_index]

plt.scatter(X_plot, y_plot)
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_plot)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y_plot)
    y_curve = poly_model.predict(X_poly)
    plt.plot(X_plot, y_curve, label="Degree " + str(degree))

plt.xlabel("Head Size(cm^3)")
plt.ylabel("Brain Weight(grams)")
plt.title("Polynomial Regression")
plt.legend()
plt.show()