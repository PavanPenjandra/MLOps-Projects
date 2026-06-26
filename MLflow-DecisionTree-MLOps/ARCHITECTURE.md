# Architecture Diagram

This diagram shows the main components of the MLflow Decision Tree MLOps project.

```text
             +----------------------+         +--------------------+
             |   Iris Dataset       |         |   MLflow Tracking  |
             |  (sklearn load_iris) |         |   Local backend    |
             +----------+-----------+         +---------+----------+
                        |                             ^
                        |                             |
                        v                             |
    +---------------------------+    +----------------+------------------+
    |  run_pipeline.py          |    |  MLflow experiment, run, model   |
    |                           |    |  metrics, params, artifacts      |
    |  1. load_data()           |    |                                  |
    |  2. train_test_split()    |    |                                  |
    |  3. fit(Pipeline)         |---->  mlflow.log_param()              |
    |  4. evaluate_model()      |    |  mlflow.log_metric()             |
    |  5. mlflow.sklearn.log_model()| |  mlflow.log_text()               |
    +---------------------------+    +----------------------------------+
                        |
                        |
                        v
    +---------------------------------------------+
    |  Model Artifact: Pipeline                    |
    |  (preprocessing + DecisionTreeClassifier)   |
    +---------------------------------------------+
                        |
                        v
    +----------------------+          +-----------------------+
    |  predict.py          |          |  serve.py             |
    |  Load model from     |          |  Load model from      |
    |  runs:/<run_id>/model|          |  runs:/<run_id>/model |
    |  Run sample inference|          |  FastAPI /predict     |
    +-----------+----------+          +-----------+-----------+
                |                                 |
                v                                 v
       +-------------------+             +-------------------+
       |  Example CLI      |             |  REST client      |
       |  output           |             |  request          |
       +-------------------+             +-------------------+
```

## Components

- `run_pipeline.py`: The main training pipeline. Loads data, trains a Decision Tree pipeline, evaluates metrics, and logs everything to MLflow.
- `src/data/load_data.py`: Loads the Iris dataset and returns feature matrix, target labels, and class names.
- `src/features/preprocess.py`: Builds preprocessing pipeline for scaling numeric features.
- `src/models/train.py`: Creates the `DecisionTreeClassifier` model.
- `src/models/evaluate.py`: Computes classification metrics.
- `predict.py`: Loads an MLflow logged model and runs local sample predictions.
- `serve.py`: Serves the MLflow logged model using FastAPI.
- `mlruns/`: Local MLflow tracking store created after training.

## Data flow

1. `run_pipeline.py` loads Iris data and splits it into train/test sets.
2. It builds a scikit-learn pipeline: preprocessing + Decision Tree model.
3. The pipeline is trained and evaluated.
4. Training metrics and the model artifact are logged to MLflow.
5. `predict.py` and `serve.py` later load the same model artifact from the MLflow run.
