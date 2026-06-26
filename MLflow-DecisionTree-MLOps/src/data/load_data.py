from sklearn.datasets import load_iris


def load_data():
    iris = load_iris(as_frame=True)
    X = iris.frame.drop(columns="target")
    y = iris.frame["target"]
    return X, y, iris.target_names
