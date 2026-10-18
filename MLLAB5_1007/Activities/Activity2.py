import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and preprocess images
X = []
y = []

folders = [("no", 0), ("yes", 1)]

for folder, label in folders:
    path = os.path.join("brain_tumor_dataset", folder)

    for file in os.listdir(path):
        img_path = os.path.join(path, file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = cv2.resize(img, (64, 64))
            X.append(img.flatten())
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset Shape:", X.shape)

# Split the data (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(X_train, y_train)

# Predictions
y_pred = svm.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Tumor", "Tumor"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display a few test images with actual and predicted labels
for i in range(min(5, len(X_test))):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap="gray")
    plt.title(
        "Actual: " + str(y_test[i]) +
        "\nPredicted: " + str(y_pred[i])
    )
    plt.axis("off")

plt.tight_layout()
plt.show()
