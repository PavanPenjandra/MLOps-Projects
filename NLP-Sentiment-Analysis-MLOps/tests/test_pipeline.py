"""
Unit tests for sentiment analysis model.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.preprocess import TextPreprocessor, FeatureEngineer, DataValidator
from data.load_data import create_train_test_split
from utils.utils import flatten_dict


class TestTextPreprocessor:
    """Test text preprocessing."""

    def test_clean_text(self):
        """Test text cleaning."""
        preprocessor = TextPreprocessor()

        text = "Hello! Check out http://example.com and email@test.com. #ML #AI 123"
        cleaned = preprocessor.clean_text(text)

        assert "http" not in cleaned
        assert "@" not in cleaned
        assert "#" not in cleaned
        assert cleaned.islower()

    def test_tokenize_text(self):
        """Test tokenization."""
        preprocessor = TextPreprocessor()
        text = "This is a test"
        tokens = preprocessor.tokenize_text(text)

        assert len(tokens) == 4
        assert tokens[0] == "This"


class TestFeatureEngineer:
    """Test feature engineering."""

    @pytest.fixture
    def feature_engineer(self):
        return FeatureEngineer(model_name="bert-base-uncased")

    def test_label_encoding(self, feature_engineer):
        """Test label encoding."""
        labels = ["positive", "negative", "positive"]
        encoded = feature_engineer.encode_labels(labels)

        assert len(encoded) == 3
        assert all(isinstance(x, (int, np.integer)) for x in encoded)

    def test_label_decoding(self, feature_engineer):
        """Test label decoding."""
        labels = ["positive", "negative", "positive"]
        encoded = feature_engineer.encode_labels(labels)
        decoded = feature_engineer.decode_labels(encoded)

        assert list(decoded) == labels

    def test_compute_text_statistics(self, feature_engineer):
        """Test text statistics computation."""
        texts = ["short", "this is a longer text", "another one"]
        stats = feature_engineer.compute_text_statistics(texts)

        assert len(stats) == 3
        assert list(stats.columns) == ["text_length", "word_count", "avg_word_length"]
        assert stats["word_count"].min() >= 1


class TestDataValidator:
    """Test data validation."""

    def test_check_class_balance(self):
        """Test class balance checking."""
        df = pd.DataFrame({"label": ["pos"] * 70 + ["neg"] * 30})

        balance = DataValidator.check_class_balance(df, "label")

        assert balance["pos"] == pytest.approx(0.7, abs=0.01)
        assert balance["neg"] == pytest.approx(0.3, abs=0.01)

    def test_validate_text_column(self):
        """Test text column validation."""
        texts = ["short", "medium length text", "a" * 100]
        report = DataValidator.validate_text_column(texts)

        assert report["total_samples"] == 3
        assert report["valid_samples"] == 3
        assert report["min_text_length"] == len("short")


class TestDataSplit:
    """Test data splitting."""

    def test_train_test_split(self):
        """Test train/test split."""
        df = pd.DataFrame(
            {"text": [f"text {i}" for i in range(100)], "label": ["pos", "neg"] * 50}
        )

        train_df, test_df = create_train_test_split(
            df, text_column="text", label_column="label", test_size=0.2
        )

        assert len(train_df) == 80
        assert len(test_df) == 20
        assert len(train_df) + len(test_df) == len(df)


class TestUtils:
    """Test utility functions."""

    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}

        flattened = flatten_dict(nested)

        assert flattened["a"] == 1
        assert flattened["b_c"] == 2
        assert flattened["b_d_e"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
