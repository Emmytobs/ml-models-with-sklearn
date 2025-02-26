from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target

print(f"Descriptive features: {X[:10]}")
print(f"Target features: {y[:10]}")