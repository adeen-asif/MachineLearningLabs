import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("headbrain.csv")
df.head()

X = df['Head Size(cm^3)'].values
y = df['Brain Weight(grams)'].values

X_sum = np.sum(X)
X_squared_sum = np.sum(X * X)
n = len(X)

print("Sum of X:", X_sum)
print("Sum of X squared:", X_squared_sum)
print("Length of X:", n)

n = len(X)

numer = n * np.sum(X * y) - np.sum(X) * np.sum(y)
denom = n * np.sum(X * X) - (np.sum(X)) ** 2
w1 = numer / denom

w0 = (np.sum(y) - w1 * np.sum(X)) / n

w1, w0

max_x = np.max(X)
min_x = np.min(X)

x1 = np.linspace(min_x, max_x)
y1 = w0 + w1 * x1

plt.plot(x1, y1, color='red', label='Regression Line')

plt.scatter(X, y, c='green', label='Scatter Plot')
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

# Calculating Root Mean Squares Error
rmse = 0
for i in range(n):
    y_pred = w0 + w1 * X[i]
    rmse += (y[i] - y_pred) ** 2

rmse = np.sqrt(rmse / n)
print("RMSE=", rmse)

ss_tot = 0
ss_res = 0
y_mean = np.mean(y)

for i in range(n):
    y_pred = w0 + w1 * X[i]
    ss_tot += (y[i] - y_mean) ** 2
    ss_res += (y[i] - y_pred) ** 2

r2 = 1 - (ss_res / ss_tot)
print("R2 Score=", r2)