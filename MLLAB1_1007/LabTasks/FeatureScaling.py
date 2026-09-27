import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = pd.DataFrame({
    "Age": [22, 25, 47, 35, 60],
    "Income": [25000, 32000, 95000, 60000, 150000]
})

# Standardization
standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(data)

# Normalization (Min-Max Scaling)
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)

# Robust Scaling
robust_scaler = RobustScaler()
data_robust = robust_scaler.fit_transform(data)

print(data_standard)
print(data_minmax)
print(data_robust)