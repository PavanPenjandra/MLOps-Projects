import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load_data import load_data
from src.features.preprocess import build_preprocessing_pipeline
from src.models.train import build_model
from src.models.evaluate import evaluate_model


def run_pipeline(experiment: str, max_depth: int, test_size: float, random_state: int):
    project_root = Path(__file__).resolve().parent
    mlruns_path = (project_root / "mlruns").as_uri()
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(experiment)

    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline(
        [
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", build_model(max_depth=max_depth, random_state=random_state)),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = evaluate_model(y_test, predictions)

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_text(
            f"target_names: {list(target_names)}\n",
            artifact_file="target_names.txt",
        )

        print("✅ Training complete")
        print(f"➡️  MLflow run id: {run.info.run_id}")
        print(f"➡️  Model URI: runs:/{run.info.run_id}/model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a decision tree and log with MLflow")
    parser.add_argument("--experiment", type=str, default="decision_tree_iris")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(
        experiment=args.experiment,
        max_depth=args.max_depth,
        test_size=args.test_size,
        random_state=args.random_state,
    )
