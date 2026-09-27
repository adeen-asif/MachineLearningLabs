from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Data
features_df = pd.DataFrame({
    'Regionname': ['North', 'South', 'East', 'West', 'North']
})

# Apply Label Encoding
le = LabelEncoder()
features_df['Region'] = le.fit_transform(features_df['Regionname'])

print("\nLabel Encoded Regions:")
print(features_df.value_counts())

