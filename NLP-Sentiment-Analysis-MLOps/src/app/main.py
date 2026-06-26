"""
FastAPI application for serving sentiment analysis model.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow

from ..serving.inference import SentimentAnalysisInference

logger = logging.getLogger(__name__)

# Global inference engine
inference_engine = None


class TextInput(BaseModel):
    """Input model for prediction."""

    text: str


class BatchTextInput(BaseModel):
    """Input model for batch predictions."""

    texts: List[str]


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    text: str
    label: str
    confidence: float
    probabilities: dict


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[PredictionResponse]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global inference_engine

    model_path = os.getenv("MODEL_PATH", "models/sentiment-model")

    logger.info(f"Loading model from {model_path}")
    inference_engine = SentimentAnalysisInference(model_path=model_path)
    logger.info("Model loaded successfully")

    yield

    logger.info("Shutting down application")


app = FastAPI(
    title="Sentiment Analysis API",
    description="NLP API for sentiment analysis",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": inference_engine is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    """
    Make prediction on single text.

    Args:
        input_data: TextInput with text field

    Returns:
        Prediction with label and confidence
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = inference_engine.predict_single(input_data.text)
        return PredictionResponse(
            text=result["text"],
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchTextInput):
    """
    Make predictions on batch of texts.

    Args:
        input_data: BatchTextInput with texts list

    Returns:
        Batch predictions
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(input_data.texts) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 100")

    try:
        results = inference_engine.predict(input_data.texts)
        predictions = [
            PredictionResponse(
                text=r["text"],
                label=r["label"],
                confidence=r["confidence"],
                probabilities=r["probabilities"],
            )
            for r in results
        ]
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain_prediction(input_data: TextInput):
    """
    Get explanation for prediction.

    Args:
        input_data: TextInput with text field

    Returns:
        Prediction explanation
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        explanation = inference_engine.explain_prediction(input_data.text)
        return explanation
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Get information about loaded model."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model": "sentiment-analysis",
        "labels": inference_engine.id2label,
        "device": inference_engine.device,
        "model_type": inference_engine.model.config.model_type,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
