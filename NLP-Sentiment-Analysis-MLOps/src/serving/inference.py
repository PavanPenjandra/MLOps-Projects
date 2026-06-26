"""
Inference module for serving sentiment analysis model.
"""

import logging
from typing import Dict, List, Any
import torch

logger = logging.getLogger(__name__)


class SentimentAnalysisInference:
    """Inference engine for sentiment analysis using Hugging Face transformers."""

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to saved model directory or Hugging Face model identifier (optional)
            device: Device to run inference on
        """
        self.device = device
        self.clf_pipeline = None
        device_id = 0 if device == "cuda" else -1

        # Lazy import pipeline inside try-except to handle compatibility issues
        try:
            try:
                from transformers.pipelines import pipeline as tf_pipeline
            except (ImportError, AttributeError):
                # Alternative import if pipelines module has issues
                exec("from transformers import pipeline as tf_pipeline", globals())
                tf_pipeline = globals()["tf_pipeline"]

            if model_path and model_path != "models/sentiment-model":
                # Try to load specified model
                logger.info(f"Loading model from {model_path}")
                self.clf_pipeline = tf_pipeline(
                    "sentiment-analysis", model=model_path, device=device_id
                )
                logger.info(f"Model loaded from {model_path}")
            else:
                raise ValueError("Using default model")
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}")
            logger.info(
                "Using default model: distilbert-base-uncased-finetuned-sst-2-english"
            )
            # Fall back with manual loader
            try:
                from transformers.pipelines import pipeline as tf_pipeline
            except (ImportError, AttributeError):
                exec("from transformers import pipeline as tf_pipeline", globals())
                tf_pipeline = globals()["tf_pipeline"]

            # Fall back to pre-trained sentiment analysis model
            self.clf_pipeline = tf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device_id,
            )

        logger.info(f"Inference pipeline ready on device {device}")

        # Set label mapping
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        self.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    def predict(
        self, texts: List[str], return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on texts.

        Args:
            texts: List of text strings
            return_probabilities: Whether to return confidence scores

        Returns:
            List of predictions with labels and scores
        """
        # Use pipeline for prediction
        results = self.clf_pipeline(texts)

        predictions = []
        for text, result in zip(texts, results):
            label = result["label"]  # POSITIVE or NEGATIVE
            score = result["score"]

            # Normalize to lowercase for consistency
            label_lower = label.lower()

            pred_dict = {
                "text": text,
                "label": label_lower,
                "confidence": score,
            }

            if return_probabilities:
                # For pipeline, we only get one score, so calculate inverse for other label
                other_score = 1.0 - score
                if label_lower == "positive":
                    pred_dict["probabilities"] = {
                        "positive": score,
                        "negative": other_score,
                    }
                else:
                    pred_dict["probabilities"] = {
                        "positive": other_score,
                        "negative": score,
                    }

            predictions.append(pred_dict)

        return predictions

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Make a prediction on a single text.

        Args:
            text: Text string to predict

        Returns:
            Prediction with label and score
        """
        predictions = self.predict([text])
        return predictions[0] if predictions else {}

    def explain(self, text: str) -> Dict[str, Any]:
        """
        Explain prediction for a text.

        Args:
            text: Text to explain

        Returns:
            Explanation with prediction details
        """
        pred = self.predict_single(text)
        return {
            "text": text,
            "prediction": pred.get("label"),
            "confidence": pred.get("confidence"),
            "probabilities": pred.get("probabilities"),
            "model": "distilbert-base-uncased-finetuned-sst-2-english",
        }
