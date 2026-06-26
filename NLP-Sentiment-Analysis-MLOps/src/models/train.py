"""
Model training module for sentiment analysis.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class SentimentAnalysisTrainer:
    """Trainer for sentiment analysis models."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        experiment_name: str = "sentiment-analysis",
    ):
        """
        Initialize trainer.

        Args:
            model_name: Pretrained model from HuggingFace
            num_labels: Number of classification labels
            device: Training device (cuda/cpu)
            experiment_name: MLflow experiment name
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device
        self.experiment_name = experiment_name

        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # MLflow setup
        mlflow.set_experiment(experiment_name)

    def train_epoch(
        self, train_loader: TorchDataLoader, optimizer, scheduler, epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx + 1}, "
                    f"Loss: {loss.item():.4f}, Acc: {correct/total:.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def evaluate(self, eval_loader: TorchDataLoader) -> Dict[str, float]:
        """
        Evaluate model.

        Args:
            eval_loader: Evaluation data loader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )

                total_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            "loss": total_loss / len(eval_loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(
                all_labels, all_preds, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                all_labels, all_preds, average="weighted", zero_division=0
            ),
            "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        }

        return metrics

    def train(
        self,
        train_loader: TorchDataLoader,
        eval_loader: TorchDataLoader,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
    ) -> Dict[str, Any]:
        """
        Complete training loop.

        Args:
            train_loader: Training data loader
            eval_loader: Evaluation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler

        Returns:
            Training history and best metrics
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "eval_loss": [],
            "eval_accuracy": [],
            "eval_f1": [],
        }

        best_f1 = 0
        best_model_path = None

        mlflow.start_run(run_name="sentiment-analysis-training")

        try:
            for epoch in range(num_epochs):
                logger.info(f"Training epoch {epoch + 1}/{num_epochs}")

                # Train
                train_metrics = self.train_epoch(
                    train_loader, optimizer, scheduler, epoch + 1
                )

                # Evaluate
                eval_metrics = self.evaluate(eval_loader)

                # Update history
                history["train_loss"].append(train_metrics["loss"])
                history["train_accuracy"].append(train_metrics["accuracy"])
                history["eval_loss"].append(eval_metrics["loss"])
                history["eval_accuracy"].append(eval_metrics["accuracy"])
                history["eval_f1"].append(eval_metrics["f1"])

                # Log to MLflow
                mlflow.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "eval_loss": eval_metrics["loss"],
                        "eval_accuracy": eval_metrics["accuracy"],
                        "eval_f1": eval_metrics["f1"],
                    },
                    step=epoch,
                )

                logger.info(
                    f"Epoch {epoch + 1}: Train Loss: {train_metrics['loss']:.4f}, "
                    f"Eval F1: {eval_metrics['f1']:.4f}"
                )

                # Save best model
                if eval_metrics["f1"] > best_f1:
                    best_f1 = eval_metrics["f1"]
                    best_model_path = f"best_model_epoch_{epoch + 1}"
                    self.model.save_pretrained(best_model_path)
                    self.tokenizer.save_pretrained(best_model_path)
                    logger.info(f"Saved best model with F1: {best_f1:.4f}")

            # Log parameters and model
            mlflow.log_params(
                {
                    "model_name": self.model_name,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "num_labels": self.num_labels,
                }
            )

            mlflow.pytorch.log_model(self.model, "model")

        finally:
            mlflow.end_run()

        return {
            "history": history,
            "best_f1": best_f1,
            "best_model_path": best_model_path,
            "final_metrics": eval_metrics,
        }

    def save_model(self, output_dir: str) -> None:
        """Save model and tokenizer."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def load_model(self, model_dir: str) -> None:
        """Load pretrained model and tokenizer."""
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = self.model.to(self.device)
        logger.info(f"Model loaded from {model_dir}")
