import os
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
path = kagglehub.dataset_download("zalando-research/fashionmnist")
fashion = pd.read_csv(os.path.join(path, "fashion-mnist_train.csv"))

X = fashion.drop(columns=["label"])
y = fashion["label"]
num_features = X.shape[1]

# 2. Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit PCA Models
# Method 1: Fixed components
pca_50 = PCA(n_components=50)
X_pca_50 = pca_50.fit_transform(X_scaled)

# Method 2: Target variance
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)

# Method 3: All components
pca_all = PCA(n_components=None)
X_pca_all = pca_all.fit_transform(X_scaled)

# 4. Generate Comparison Report
comparison = pd.DataFrame(
    {
        "Method": [
            "Fixed Components (50)",
            "Variance Target (95%)",
            "All Components",
        ],
        "Original Features": [num_features] * 3,
        "Retained Components": [
            pca_50.n_components_,
            pca_95.n_components_,
            pca_all.n_components_,
        ],
        "Explained Variance": [
            np.sum(pca_50.explained_variance_ratio_),
            np.sum(pca_95.explained_variance_ratio_),
            np.sum(pca_all.explained_variance_ratio_),
        ],
    }
)

print(comparison.to_string(index=False))
print(
    f"\nGreatest reduction: {comparison.loc[comparison['Retained Components'].idxmin(), 'Method']}"
)
print(
    f"Max info retained: {comparison.loc[comparison['Explained Variance'].idxmax(), 'Method']}\n"
)

# 5. Plot: Cumulative Explained Variance
cumsum_all = np.cumsum(pca_all.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumsum_all) + 1), cumsum_all, color="darkblue", lw=2)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Variance vs. Components (Fashion MNIST)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# 6. Plot: 2D Projection Space
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_pca_all[:, 0],
    X_pca_all[:, 1],
    c=y,
    cmap="tab10",
    edgecolors="none",
    alpha=0.5,
)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA Projection Map (Fashion MNIST)")

cbar = plt.colorbar(scatter, ticks=range(10), label="Clothing Class")
cbar.ax.set_yticklabels(class_names)
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()