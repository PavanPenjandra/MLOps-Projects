import argparse
from typing import List

import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

from src.data.load_data import load_data

app = FastAPI()
model = None
target_names = []


class PredictionRequest(BaseModel):
    feature_vector: List[float]


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Decision tree inference service is ready."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global model, target_names
    if model is None:
        raise RuntimeError("Model is not loaded.")
    prediction = model.predict([request.feature_vector])[0]
    class_name = target_names[prediction] if target_names else str(prediction)
    return PredictionResponse(prediction=int(prediction), class_name=class_name)


def main(model_uri: str, host: str = "127.0.0.1", port: int = 8000):
    global model, target_names
    model = mlflow.pyfunc.load_model(model_uri)
    _, _, target_names = load_data()
    print(f"🚀 Loaded model from {model_uri}")
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start FastAPI service for MLflow logged model")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(model_uri=f"runs:/{args.run_id}/model", host=args.host, port=args.port)
