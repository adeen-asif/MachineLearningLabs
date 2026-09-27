from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Data
features_df = pd.DataFrame({
    'Regionname': ['North', 'South', 'East', 'West', 'North']
})

# One Hot Encoding Example

# Assuming 'Method' is a categorical column
features_df['Method'] = ['Method1', 'Method2', 'Method1', 'Method3', 'Method2']

df_one_hot = pd.get_dummies(features_df['Method'])

print("\nOne Hot Encoded DataFrame:")
print(df_one_hot.value_counts())