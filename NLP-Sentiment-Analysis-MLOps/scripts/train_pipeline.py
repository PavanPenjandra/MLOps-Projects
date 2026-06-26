"""
Main training pipeline script.
"""

import logging
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoTokenizer

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import DataLoader as DataLoaderModule
from src.features.preprocess import FeatureEngineer, TextPreprocessor
from src.models.train import SentimentAnalysisTrainer
from src.utils.utils import setup_logging, load_config, create_directories

logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """PyTorch dataset for sentiment analysis."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def main(config_path: str = "config.yaml"):
    """
    Main training pipeline.

    Args:
        config_path: Path to configuration file
    """
    # Setup
    setup_logging(log_level="INFO")
    config = load_config(config_path)

    # Create directories
    create_directories(
        [
            config["paths"]["data_dir"],
            config["paths"]["model_dir"],
            config["paths"]["logs_dir"],
        ]
    )

    logger.info("Starting training pipeline")
    logger.info(f"Config: {config}")

    # 1. Load data
    logger.info("Loading data...")
    data_loader = DataLoaderModule(data_dir=config["paths"]["data_dir"])

    # Load from HuggingFace or local
    if config["data"]["source"] == "huggingface":
        df = data_loader.load_huggingface_dataset(
            dataset_name=config["data"]["dataset_name"],
            split=config["data"]["split"],
            sample_size=config["data"].get("sample_size"),
        )
    else:
        df = data_loader.load_local_csv(config["data"]["path"])

    # 2. Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = TextPreprocessor()
    df["text_clean"] = df[config["data"]["text_column"]].apply(preprocessor.clean_text)

    # 3. Train/test split
    train_df, test_df = data_loader.create_train_test_split(
        df,
        text_column="text_clean",
        label_column=config["data"]["label_column"],
        test_size=config["training"]["test_size"],
    )

    # 4. Feature engineering
    logger.info("Engineering features...")
    feature_engineer = FeatureEngineer(model_name=config["training"]["model_name"])

    # Encode labels
    train_labels = feature_engineer.encode_labels(
        train_df[config["data"]["label_column"]].tolist()
    )
    test_labels = feature_engineer.encode_labels(
        test_df[config["data"]["label_column"]].tolist()
    )

    # 5. Create datasets and dataloaders
    logger.info("Creating data loaders...")
    tokenizer = AutoTokenizer.from_pretrained(config["training"]["model_name"])

    train_dataset = SentimentDataset(
        texts=train_df["text_clean"].tolist(),
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=config["training"]["max_length"],
    )

    test_dataset = SentimentDataset(
        texts=test_df["text_clean"].tolist(),
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=config["training"]["max_length"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # 6. Train model
    logger.info("Training model...")
    trainer = SentimentAnalysisTrainer(
        model_name=config["training"]["model_name"],
        num_labels=len(feature_engineer.label_encoder.classes_),
        experiment_name=config["training"]["experiment_name"],
    )

    results = trainer.train(
        train_loader=train_loader,
        eval_loader=test_loader,
        num_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"].get("warmup_steps", 0),
    )

    logger.info(f"Training complete. Best F1: {results['best_f1']:.4f}")

    # 7. Save model
    logger.info("Saving model...")
    model_output_path = Path(config["paths"]["model_dir"]) / "sentiment-model"
    trainer.save_model(str(model_output_path))

    # Save label encoder
    import pickle

    encoder_path = model_output_path / "label_encoder.pkl"
    with open(encoder_path, "wb") as f:
        pickle.dump(feature_engineer.label_encoder, f)

    logger.info(f"Model saved to {model_output_path}")
    logger.info("Training pipeline complete!")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    results = main(config_path=args.config)
