"""
pytest configuration and fixtures.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This movie was absolutely fantastic!",
        "I hated every minute of it",
        "It was okay, nothing special",
        "Best decision I made this year",
        "Complete waste of time",
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return ["positive", "negative", "neutral", "positive", "negative"]


@pytest.fixture
def sample_dataframe(sample_texts, sample_labels):
    """Sample DataFrame for testing."""
    import pandas as pd

    return pd.DataFrame({"text": sample_texts, "label": sample_labels})
