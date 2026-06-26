"""
Data quality validation module.
"""

import logging
from typing import Dict, List, Any
import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Validate data quality for production."""

    @staticmethod
    def validate_production_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate production data meets quality standards.

        Args:
            df: Input DataFrame

        Returns:
            Validation report
        """
        report = {"passed": True, "issues": [], "warnings": []}

        # Check for empty DataFrame
        if df.empty:
            report["passed"] = False
            report["issues"].append("DataFrame is empty")
            return report

        # Check for required columns
        if "text" not in df.columns or "label" not in df.columns:
            report["passed"] = False
            report["issues"].append("Missing required columns: text, label")

        # Check for missing values
        missing_text = df["text"].isnull().sum()
        if missing_text > 0:
            report["issues"].append(f"{missing_text} missing text values")
            report["passed"] = False

        # Check text length
        df["text_length"] = df["text"].str.len()
        short_texts = (df["text_length"] < 5).sum()
        if short_texts > 0:
            report["warnings"].append(f"{short_texts} texts shorter than 5 characters")

        # Check class balance
        label_dist = df["label"].value_counts(normalize=True)
        min_class_pct = label_dist.min() * 100
        if min_class_pct < 10:
            report["warnings"].append(
                f"Class imbalance detected: smallest class is {min_class_pct:.1f}%"
            )

        # Check for duplicates
        duplicates = df["text"].duplicated().sum()
        if duplicates > 0:
            report["warnings"].append(f"{duplicates} duplicate texts found")

        return report

    @staticmethod
    def check_drift(baseline_stats: Dict, current_stats: Dict) -> Dict[str, Any]:
        """
        Check for data drift between baseline and current data.

        Args:
            baseline_stats: Statistics from baseline data
            current_stats: Statistics from current data

        Returns:
            Drift analysis report
        """
        report = {"drift_detected": False, "metrics": {}}

        # Compare text length
        baseline_len_mean = baseline_stats["text_length_mean"]
        current_len_mean = current_stats["text_length_mean"]
        pct_change = abs(current_len_mean - baseline_len_mean) / baseline_len_mean * 100

        report["metrics"]["text_length_drift"] = pct_change

        if pct_change > 10:  # Threshold
            report["drift_detected"] = True
            logger.warning(f"Text length drift detected: {pct_change:.1f}%")

        return report
