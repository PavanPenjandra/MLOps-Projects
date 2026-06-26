"""
Model evaluation module.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import torch

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate sentiment analysis models."""

    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities

        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # Classification report
        metrics["classification_report"] = classification_report(
            y_true, y_pred, output_dict=True
        )

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        # ROC-AUC if binary classification
        if len(np.unique(y_true)) == 2 and y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics["roc_auc"] = None

        return metrics

    @staticmethod
    def check_model_performance(
        metrics: Dict[str, Any], thresholds: Dict[str, float]
    ) -> Tuple[bool, list]:
        """
        Check if model meets performance thresholds.

        Args:
            metrics: Evaluation metrics
            thresholds: Performance thresholds

        Returns:
            Tuple of (passes_all, failures)
        """
        failures = []

        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                if metrics[metric_name] < threshold:
                    failures.append(
                        f"{metric_name}: {metrics[metric_name]:.4f} < {threshold}"
                    )

        passes_all = len(failures) == 0
        return passes_all, failures

    @staticmethod
    def log_eval_report(metrics: Dict[str, Any]) -> None:
        """Log evaluation report."""
        logger.info("=" * 50)
        logger.info("MODEL EVALUATION REPORT")
        logger.info("=" * 50)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        if "roc_auc" in metrics and metrics["roc_auc"]:
            logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info("=" * 50)
