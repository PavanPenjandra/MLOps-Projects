from sklearn.tree import DecisionTreeClassifier


def build_model(max_depth: int = 3, random_state: int = 42):
    return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
