from src.models.train import build_model
from src.models.evaluate import evaluate_model


def test_build_model_returns_classifier():
    model = build_model(max_depth=3, random_state=42)
    assert model.max_depth == 3
    assert model.random_state == 42


def test_evaluate_model_metrics():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    metrics = evaluate_model(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 0.75
