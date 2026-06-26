"""
Example inference script showing different usage patterns.
"""

import logging
from pathlib import Path
from serving.inference import SentimentAnalysisInference

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_single_prediction():
    """Example: Single text prediction."""
    logger.info("\n=== Example 1: Single Prediction ===")

    inference = SentimentAnalysisInference("models/sentiment-model")

    text = "This movie was absolutely fantastic!"
    result = inference.predict_single(text)

    logger.info(f"Text: {text}")
    logger.info(f"Prediction: {result['label']}")
    logger.info(f"Confidence: {result['confidence']:.2%}")
    logger.info(f"Probabilities: {result['probabilities']}")


def example_batch_prediction():
    """Example: Batch text predictions."""
    logger.info("\n=== Example 2: Batch Prediction ===")

    inference = SentimentAnalysisInference("models/sentiment-model")

    texts = [
        "I absolutely love this product!",
        "This is the worst experience ever",
        "It's okay, nothing special",
        "Highly recommended to everyone",
        "Complete waste of money",
    ]

    results = inference.predict(texts)

    for result in results:
        logger.info(
            f"{result['text'][:30]}... -> "
            f"{result['label']} ({result['confidence']:.2%})"
        )


def example_large_batch():
    """Example: Large batch with streaming."""
    logger.info("\n=== Example 3: Large Batch Processing ===")

    inference = SentimentAnalysisInference("models/sentiment-model")

    # Simulate 500 texts
    texts = [f"Sample text number {i}" for i in range(500)]

    results = inference.batch_predict(texts, batch_size=32)

    logger.info(f"Processed {len(results)} texts")
    logger.info(f"Positive: {sum(1 for r in results if r['label'] == 'positive')}")
    logger.info(f"Negative: {sum(1 for r in results if r['label'] == 'negative')}")


def example_explanation():
    """Example: Get prediction explanation."""
    logger.info("\n=== Example 4: Prediction Explanation ===")

    inference = SentimentAnalysisInference("models/sentiment-model")

    text = "I really enjoyed this amazing product!"
    explanation = inference.explain_prediction(text)

    logger.info(f"Text: {text}")
    logger.info(f"Prediction: {explanation['prediction']}")
    logger.info(f"Confidence: {explanation['confidence']:.2%}")
    logger.info(f"Tokens: {explanation['tokens']}")


def example_with_api():
    """Example: Using FastAPI client."""
    logger.info("\n=== Example 5: FastAPI Usage ===")

    import requests

    base_url = "http://localhost:8000"

    # Single prediction
    response = requests.post(f"{base_url}/predict", json={"text": "Great service!"})
    logger.info(f"Single prediction: {response.json()}")

    # Batch prediction
    response = requests.post(
        f"{base_url}/predict_batch", json={"texts": ["Good", "Bad", "Neutral"]}
    )
    logger.info(f"Batch predictions: {len(response.json()['predictions'])} results")

    # Health check
    response = requests.get(f"{base_url}/health")
    logger.info(f"Health status: {response.json()}")


if __name__ == "__main__":
    # Check if model exists
    model_path = Path("models/sentiment-model")

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train the model first: python scripts/train_pipeline.py")
        exit(1)

    # Run examples
    try:
        example_single_prediction()
        example_batch_prediction()
        example_large_batch()
        example_explanation()
        logger.info("\n✓ All examples completed successfully!")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
