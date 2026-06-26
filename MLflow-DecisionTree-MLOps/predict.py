import argparse

import mlflow
from src.data.load_data import load_data


def predict_from_run(run_id: str, sample_count: int = 3):
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    X, y, target_names = load_data()

    sample = X.head(sample_count)
    predictions = model.predict(sample)
    print(f"Loaded model from {model_uri}")
    print("Sample predictions:")
    for row_index, (features, prediction) in enumerate(zip(sample.values, predictions), start=1):
        print(f"  {row_index}: features={list(features)}, prediction={target_names[prediction]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load an MLflow model run and run sample inference")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--sample_count", type=int, default=3)
    args = parser.parse_args()
    predict_from_run(args.run_id, sample_count=args.sample_count)
