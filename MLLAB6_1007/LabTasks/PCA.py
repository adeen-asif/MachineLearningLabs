import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()

X = wine.data
Y = wine.target
feat_names = wine.feature_names
target_names = wine.target_names

print("Original dataset shape:", X.shape)
print("Target shape:", Y.shape)
print("Features names:", feat_names)
print("Target names:", target_names)


scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

print("Original dataset shape:", X.shape)
print("Reduced dataset shape:", X_reduced.shape)

wine_colors = ['orangered', 'darkred', 'crimson']

plt.figure(figsize=(10,6))

for class_num, color, target_name in zip (
    [0,1,2], wine_colors, target_names
):
    plt.scatter(
        X_reduced[Y == class_num, 0],
        X_reduced[Y == class_num, 1],
        color = color,
        label = target_name,
        alpha = 0.7
    )

plt.title('PCA of Wine Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(alpha=0.2)
plt.show()

print('Principal component loadings:')
print(pca.components_)
print("Shape of principal components:", pca.components_.shape)

plt.figure(figsize=(14, 5))
sns.heatmap(
    pca.components_,
    cmap="viridis",
    xticklabels=feat_names,
    yticklabels=["First Component", "Second Component"]
)
plt.title("PCA Component Loadings")
plt.xlabel("Feature Names")
plt.ylabel("Principal Components")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

pca_3 = PCA(n_components=0.96)
X_reduced_96 = pca_3.fit_transform(X_scaled)

print("Number of components selected:", pca_3.n_components_)
print("Total explained variance:", np.sum(pca_3.explained_variance_ratio_))
print("Reduced dataset shape:", X_reduced_96.shape)