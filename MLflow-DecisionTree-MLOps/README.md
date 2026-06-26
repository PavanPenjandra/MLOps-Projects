# MLflow Decision Tree MLOps Project

A self-contained end-to-end MLOps demo using `scikit-learn` Decision Tree classification and MLflow experiment tracking.

## What is included

- Data loading using the Iris dataset
- Preprocessing and training with a `DecisionTreeClassifier`
- MLflow experiment tracking for parameters, metrics, artifacts, and model versions
- Inference script for loading a logged MLflow model
- FastAPI service for production-style REST inference

## Setup

```powershell
cd MLflow-DecisionTree-MLOps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train and log a model

```powershell
python run_pipeline.py --experiment decision_tree_iris --max_depth 3
```

## Run MLflow UI

```powershell
mlflow ui --backend-store-uri file:./mlruns
```

Open http://127.0.0.1:5000 to inspect experiments, runs, metrics, and model artifacts.

## Inference

```powershell
python predict.py --run_id <RUN_ID>
```

## Serve as REST API

```powershell
python serve.py --run_id <RUN_ID>
```

Then request:

```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"feature_vector\":[5.1,3.5,1.4,0.2]}"
```

## Project structure

- `run_pipeline.py` - train + MLflow logging pipeline
- `predict.py` - load a logged MLflow model and run example predictions
- `serve.py` - FastAPI inference service
- `src/data/load_data.py` - data loader
- `src/features/preprocess.py` - preprocessing utilities
- `src/models/train.py` - decision tree model factory
- `src/models/evaluate.py` - metric evaluation helpers
- `mlruns/` - local MLflow tracking store (generated after training)
