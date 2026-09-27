from sklearn.datasets import fetch_olivetti_faces

# fetch the faces data
faces = fetch_olivetti_faces()
faces.keys()

n_samples, n_features = faces.data.shape
print((n_samples, n_features))
print(faces.images.shape)
print(faces.data.shape)
print(faces.DESCR)