from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt

# fetch the faces data
faces = fetch_olivetti_faces()
faces.keys()

n_samples, n_features = faces.data.shape
print((n_samples, n_features))
print(faces.images.shape)
print(faces.data.shape)
print(faces.DESCR)

# Show first 16 images
plt.figure(figsize=(6, 6))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(faces.images[i], cmap="gray")
    plt.axis("off") # hide axes
    plt.title(faces.target[i])

plt.show()