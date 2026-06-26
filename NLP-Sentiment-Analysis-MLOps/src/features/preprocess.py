"""
Text preprocessing and feature engineering for NLP.
"""

import logging
import re
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text string

        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www.\S+", "", text)

        # Remove emails
        text = re.sub(r"\S+@\S+", "", text)

        # Remove special characters and digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def remove_stopwords(text: str, stopwords: set) -> str:
        """Remove stopwords from text."""
        words = text.split()
        filtered = [w for w in words if w.lower() not in stopwords]
        return " ".join(filtered)

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """Simple word tokenization."""
        return text.split()


class FeatureEngineer:
    """Feature engineering for NLP."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize feature engineer.

        Args:
            model_name: Transformer model name from HuggingFace
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """Encode text labels to integers."""
        return self.label_encoder.fit_transform(labels)

    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """Decode integer labels back to text."""
        return self.label_encoder.inverse_transform(encoded_labels)

    def tokenize_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> Dict[str, Any]:
        """
        Tokenize batch of texts using transformer tokenizer.

        Args:
            texts: List of text strings
            max_length: Maximum token length
            padding: Padding strategy
            truncation: Whether to truncate long texts

        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="np",
        )
        return encodings

    def compute_text_statistics(self, texts: List[str]) -> pd.DataFrame:
        """
        Compute text statistics.

        Args:
            texts: List of texts

        Returns:
            DataFrame with statistics
        """
        stats = {
            "text_length": [len(t) for t in texts],
            "word_count": [len(t.split()) for t in texts],
            "avg_word_length": [
                np.mean([len(w) for w in t.split()]) if t.split() else 0 for t in texts
            ],
        }
        return pd.DataFrame(stats)


class DataValidator:
    """Validate data quality."""

    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
        """Check for missing values."""
        missing = df.isnull().sum()
        return missing[missing > 0].to_dict()

    @staticmethod
    def check_class_balance(df: pd.DataFrame, label_col: str) -> pd.Series:
        """Check class distribution."""
        return df[label_col].value_counts(normalize=True)

    @staticmethod
    def validate_text_column(
        texts: List[str], min_length: int = 1, max_length: int = 10000
    ) -> Dict[str, Any]:
        """
        Validate text column.

        Args:
            texts: List of texts
            min_length: Minimum text length
            max_length: Maximum text length

        Returns:
            Validation report
        """
        valid_count = sum(1 for t in texts if min_length <= len(t) <= max_length)

        return {
            "total_samples": len(texts),
            "valid_samples": valid_count,
            "invalid_samples": len(texts) - valid_count,
            "min_text_length": min(len(t) for t in texts) if texts else 0,
            "max_text_length": max(len(t) for t in texts) if texts else 0,
            "avg_text_length": np.mean([len(t) for t in texts]) if texts else 0,
        }
