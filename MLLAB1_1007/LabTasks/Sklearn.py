from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()

# Show feature data (x)
print("Features (first 5 rows):")
print(iris.data[:5])

# Show target values (y)
print("\nTarget values (first 5):")
print(iris.target[:5])
