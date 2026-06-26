"""
Data quality checks using Great Expectations (if available) or custom validators.
Validates data before and after processing.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import great_expectations as ge
    from great_expectations.core.batch import RuntimeBatchRequest

    GREAT_EXPECTATIONS_AVAILABLE = True
except ImportError:
    GREAT_EXPECTATIONS_AVAILABLE = False
    logger.warning("Great Expectations not available. Using basic validation.")


@dataclass
class DataQualityCheckResult:
    """Result of a data quality check."""

    check_name: str
    passed: bool
    details: Dict
    timestamp: str


class DataQualityValidator:
    """
    Custom data quality validator with Great Expectations support.
    Performs checks on text data and model inputs.
    """

    def __init__(self, use_great_expectations: bool = True):
        """
        Initialize validator.

        Args:
            use_great_expectations: Use Great Expectations if available
        """
        self.use_ge = use_great_expectations and GREAT_EXPECTATIONS_AVAILABLE
        self.checks_passed = 0
        self.checks_failed = 0
        self.results = []

    def check_text_length(
        self, texts: List[str], min_length: int = 1, max_length: int = 512
    ) -> DataQualityCheckResult:
        """
        Check text length constraints.

        Args:
            texts: List of texts
            min_length: Minimum length
            max_length: Maximum length

        Returns:
            Check result
        """
        from datetime import datetime

        invalid_texts = []
        for i, text in enumerate(texts):
            if len(text) < min_length or len(text) > max_length:
                invalid_texts.append(
                    {
                        "index": i,
                        "length": len(text),
                        "text": text[:50] + "..." if len(text) > 50 else text,
                    }
                )

        passed = len(invalid_texts) == 0
        result = DataQualityCheckResult(
            check_name="text_length",
            passed=passed,
            details={
                "total_texts": len(texts),
                "invalid_count": len(invalid_texts),
                "min_length": min_length,
                "max_length": max_length,
                "invalid_examples": invalid_texts[:5],
            },
            timestamp=datetime.now().isoformat(),
        )

        self._record_result(result)
        return result

    def check_null_values(self, texts: List[str]) -> DataQualityCheckResult:
        """
        Check for null or empty values.

        Args:
            texts: List of texts

        Returns:
            Check result
        """
        from datetime import datetime

        null_indices = [
            i
            for i, t in enumerate(texts)
            if t is None or (isinstance(t, str) and len(t.strip()) == 0)
        ]

        passed = len(null_indices) == 0
        result = DataQualityCheckResult(
            check_name="null_values",
            passed=passed,
            details={
                "total_texts": len(texts),
                "null_count": len(null_indices),
                "null_indices": null_indices[:10],
            },
            timestamp=datetime.now().isoformat(),
        )

        self._record_result(result)
        return result

    def check_encoding(self, texts: List[str]) -> DataQualityCheckResult:
        """
        Check text encoding.

        Args:
            texts: List of texts

        Returns:
            Check result
        """
        from datetime import datetime

        encoding_errors = []
        for i, text in enumerate(texts):
            try:
                text.encode("utf-8")
            except UnicodeEncodeError as e:
                encoding_errors.append({"index": i, "error": str(e)})

        passed = len(encoding_errors) == 0
        result = DataQualityCheckResult(
            check_name="encoding",
            passed=passed,
            details={
                "total_texts": len(texts),
                "encoding_errors": len(encoding_errors),
                "error_examples": encoding_errors[:5],
            },
            timestamp=datetime.now().isoformat(),
        )

        self._record_result(result)
        return result

    def check_language_distribution(
        self,
        texts: List[str],
        expected_languages: List[str],
        min_samples_per_language: int = 1,
    ) -> DataQualityCheckResult:
        """
        Check language distribution.

        Args:
            texts: List of texts
            expected_languages: List of expected language codes
            min_samples_per_language: Minimum samples per language

        Returns:
            Check result
        """
        from datetime import datetime

        try:
            from langdetect import detect_langs

            language_dist = {}
            for text in texts:
                try:
                    detected = detect_langs(text)[0].lang
                    language_dist[detected] = language_dist.get(detected, 0) + 1
                except:
                    pass
        except ImportError:
            language_dist = {}

        passed = all(
            language_dist.get(lang, 0) >= min_samples_per_language
            for lang in expected_languages
        )

        result = DataQualityCheckResult(
            check_name="language_distribution",
            passed=passed,
            details={
                "distribution": language_dist,
                "expected_languages": expected_languages,
                "min_per_language": min_samples_per_language,
            },
            timestamp=datetime.now().isoformat(),
        )

        self._record_result(result)
        return result

    def check_duplicate_texts(self, texts: List[str]) -> DataQualityCheckResult:
        """
        Check for duplicate texts.

        Args:
            texts: List of texts

        Returns:
            Check result
        """
        from datetime import datetime
        from collections import Counter

        text_counts = Counter(texts)
        duplicates = {text: count for text, count in text_counts.items() if count > 1}

        passed = len(duplicates) == 0
        result = DataQualityCheckResult(
            check_name="duplicate_texts",
            passed=passed,
            details={
                "total_texts": len(texts),
                "unique_texts": len(text_counts),
                "duplicate_count": len(duplicates),
                "duplicate_examples": {k: v for k, v in list(duplicates.items())[:5]},
            },
            timestamp=datetime.now().isoformat(),
        )

        self._record_result(result)
        return result

    def check_label_distribution(
        self, labels: List[str], min_samples_per_label: int = 10
    ) -> DataQualityCheckResult:
        """
        Check label distribution.

        Args:
            labels: List of labels
            min_samples_per_label: Minimum samples per label

        Returns:
            Check result
        """
        from datetime import datetime
        from collections import Counter

        label_counts = Counter(labels)
        imbalanced_labels = {
            label: count
            for label, count in label_counts.items()
            if count < min_samples_per_label
        }

        passed = len(imbalanced_labels) == 0
        result = DataQualityCheckResult(
            check_name="label_distribution",
            passed=passed,
            details={
                "label_counts": dict(label_counts),
                "imbalanced_labels": imbalanced_labels,
                "min_samples_per_label": min_samples_per_label,
            },
            timestamp=datetime.now().isoformat(),
        )

        self._record_result(result)
        return result

    def check_special_characters(
        self, texts: List[str], allowed_special_chars: str = ""
    ) -> DataQualityCheckResult:
        """
        Check for unexpected special characters.

        Args:
            texts: List of texts
            allowed_special_chars: Allowed special characters

        Returns:
            Check result
        """
        from datetime import datetime
        import re

        # Default allowed: letters, digits, spaces, common punctuation
        default_allowed = r"[a-zA-Z0-9\s\.\,\!\?\-\'\"\(\)]"
        pattern = default_allowed + allowed_special_chars

        problematic = []
        for i, text in enumerate(texts):
            # Find characters not matching pattern
            if not re.match(f"^{pattern}*$", text):
                problematic.append({"index": i, "text": text[:50]})

        passed = len(problematic) == 0
        result = DataQualityCheckResult(
            check_name="special_characters",
            passed=passed,
            details={
                "total_texts": len(texts),
                "problematic_count": len(problematic),
                "problematic_examples": problematic[:5],
            },
            timestamp=datetime.now().isoformat(),
        )

        self._record_result(result)
        return result

    def run_full_validation(
        self, texts: List[str], labels: Optional[List[str]] = None
    ) -> Dict:
        """
        Run full data quality validation.

        Args:
            texts: List of texts
            labels: List of labels (optional)

        Returns:
            Validation summary
        """
        logger.info("Starting full data quality validation...")

        results = []
        results.append(self.check_null_values(texts))
        results.append(self.check_text_length(texts))
        results.append(self.check_encoding(texts))
        results.append(self.check_duplicate_texts(texts))
        results.append(self.check_special_characters(texts))

        if labels:
            results.append(self.check_label_distribution(labels))

        summary = {
            "total_checks": len(results),
            "passed_checks": sum(1 for r in results if r.passed),
            "failed_checks": sum(1 for r in results if not r.passed),
            "results": [
                {
                    "check": r.check_name,
                    "passed": r.passed,
                    "details": r.details,
                }
                for r in results
            ],
        }

        logger.info(
            f"Validation complete: {summary['passed_checks']}/{summary['total_checks']} checks passed"
        )

        return summary

    def _record_result(self, result: DataQualityCheckResult):
        """Record check result."""
        self.results.append(result)
        if result.passed:
            self.checks_passed += 1
            logger.info(f"✅ {result.check_name}: PASSED")
        else:
            self.checks_failed += 1
            logger.warning(f"❌ {result.check_name}: FAILED - {result.details}")

    def print_validation_report(self):
        """Print validation report."""
        print(f"\n{'='*70}")
        print("📊 Data Quality Validation Report")
        print("=" * 70)
        print(f"Total Checks: {len(self.results)}")
        print(f"Passed: {self.checks_passed} ✅")
        print(f"Failed: {self.checks_failed} ❌")
        print(f"Pass Rate: {(self.checks_passed / len(self.results) * 100):.1f}%")

        print(f"\nDetails:")
        for result in self.results:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.check_name:30s}: {result.details}")

        print("=" * 70 + "\n")


class MonitoringThresholds:
    """Monitoring thresholds for production models."""

    @staticmethod
    def sentiment_analysis() -> Dict:
        """Thresholds for sentiment analysis."""
        return {
            "min_accuracy": 0.90,
            "max_error_rate": 0.10,
            "min_confidence": 0.75,
            "max_latency_ms": 100,
            "max_null_predictions": 0.05,
        }

    @staticmethod
    def multilingual() -> Dict:
        """Thresholds for multilingual models."""
        return {
            "min_accuracy_per_language": 0.85,
            "max_language_imbalance": 0.20,
            "min_language_coverage": 0.95,
            "max_translation_errors": 0.05,
        }
