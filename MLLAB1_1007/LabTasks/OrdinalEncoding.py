import pandas as pd

# Handling Categorical Attributes
# Manual Encoding Example
mapping = {'h': 1, 'u': 2, 't': 3}
features = pd.DataFrame({'Type': ['h', 'u', 't', 'h', 'u']})
features['Type'] = features['Type'].map(mapping)

print("\nEncoded Types:")
print(features.groupby('Type').size())