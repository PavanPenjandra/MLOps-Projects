from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_preprocessing_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )
