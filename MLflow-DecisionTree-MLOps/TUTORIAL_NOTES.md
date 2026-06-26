# MLflow Decision Tree MLOps Tutorial Notes

## 1. Project Overview

This project demonstrates a simple end-to-end MLOps workflow using:

- `scikit-learn` Decision Tree classification
- `MLflow` experiment tracking and model logging
- `FastAPI` for serving predictions as a REST API
- a clean modular Python package structure

The dataset is the classic Iris dataset, which is small and easy to understand.

## 2. Architecture Summary

The project is organized into three main layers:

1. Data and preprocessing
2. Model training and evaluation
3. Model persistence and serving

### Architecture flow

- `run_pipeline.py` loads the data, builds the preprocessing and training pipeline, trains a Decision Tree, evaluates the model, and logs everything to MLflow.
- `predict.py` loads a logged MLflow model from a run and runs sample predictions.
- `serve.py` starts a FastAPI service and exposes a `/predict` endpoint for online inference.

MLflow stores:

- run parameters (model type, max depth, test split, random seed)
- metrics (accuracy, precision, recall, F1 score)
- trained model artifact
- a small text artifact containing target names

## 3. File Architecture

### Top-level files

- `README.md`
  - Project description and quick start instructions.
- `requirements.txt`
  - Python package dependencies needed to run the project.
- `run_pipeline.py`
  - Main training pipeline and MLflow logging script.
- `predict.py`
  - Example inference script that loads a model from MLflow.
- `serve.py`
  - FastAPI application for serving predictions over HTTP.
- `TUTORIAL_NOTES.md`
  - Beginner-friendly explanation of the project and architecture.
- `.gitignore`
  - Files and folders excluded from version control, including `mlruns/`.

### Source package

The `src` package contains reusable code for data, features, and models.

- `src/__init__.py`
  - Marks `src` as a Python package.
- `src/data/load_data.py`
  - Loads the Iris dataset and returns features, labels, and class names.
- `src/features/preprocess.py`
  - Creates a preprocessing pipeline; here it standardizes numeric data with `StandardScaler`.
- `src/models/train.py`
  - Creates a `DecisionTreeClassifier` model with configurable `max_depth`.
- `src/models/evaluate.py`
  - Computes evaluation metrics: accuracy, precision, recall, and F1 score.

### Tests

- `tests/test_pipeline.py`
  - Contains simple unit tests for the model builder and evaluation helper.

## 4. Detailed Python File Explanations

### `run_pipeline.py`

This is the main training script.

- Sets MLflow tracking to local file storage in `mlruns/`.
- Calls `load_data()` to load Iris features and labels.
- Splits data into training and test sets.
- Builds a `Pipeline` with preprocessing and the decision tree model.
- Fits the pipeline on training data.
- Evaluates the test predictions and logs metrics.
- Logs the trained model as an MLflow artifact.
- Prints the MLflow run ID and model URI for later use.

### `predict.py`

This script demonstrates how to load a model from MLflow and make predictions.

- Accepts a `--run_id` input to load the specific MLflow run.
- Loads the model from `runs:/{run_id}/model`.
- Loads the data again to retrieve sample features and class names.
- Prints predictions for a few sample rows.

### `serve.py`

This script starts the FastAPI service.

- Defines a request schema (`PredictionRequest`) with `feature_vector`.
- Defines a response schema (`PredictionResponse`) with prediction index and class name.
- Loads the MLflow model at startup using the provided `run_id`.
- Uses the same class names from the Iris dataset to return a human-readable label.
- Exposes `/predict` for POST inference requests.

### `src/data/load_data.py`

- Loads the Iris dataset using `sklearn.datasets.load_iris`.
- Returns a dataframe of input features, the target labels, and the label names.

### `src/features/preprocess.py`

- Builds a scikit-learn `Pipeline` that applies `StandardScaler`.
- Standard scaling helps the model by centering and scaling continuous features.

### `src/models/train.py`

- Returns a configured `DecisionTreeClassifier`.
- The `max_depth` parameter controls model complexity.

### `src/models/evaluate.py`

- Computes common classification metrics.
- Uses `average="macro"` so each Iris class is weighted equally.

## 5. Beginner Walkthrough

### Step 1: Install dependencies

```powershell
cd MLflow-DecisionTree-MLOps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Train and log a model

```powershell
python run_pipeline.py --experiment decision_tree_iris --max_depth 3
```

This will create a new MLflow run and save the model under `mlruns/`.

### Step 3: Inspect MLflow results

```powershell
mlflow ui --backend-store-uri file:./mlruns
```

Then open `http://127.0.0.1:5000` in your browser.

### Step 4: Run sample predictions

```powershell
python predict.py --run_id <RUN_ID>
```

Use the run ID printed by `run_pipeline.py`.

### Step 5: Start the API service

```powershell
python serve.py --run_id <RUN_ID>
```

Send a test request:

```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"feature_vector\":[5.1,3.5,1.4,0.2]}"
```

## 6. Why this is a good beginner MLOps project

- Uses a small, well-known dataset.
- Demonstrates training, evaluation, logging, and serving.
- MLflow tracks the full experiment lifecycle.
- The code is modular and easy to extend.
- The service can be replaced with other model types without changing the overall architecture.

## 7. Next learning steps

- Try changing `max_depth` and compare MLflow metrics.
- Add a new dataset and reuse the same `src/` modules.
- Add a model registry or deployment pipeline.
- Add input validation and logging to the FastAPI service.
