"""
Multi-language text classifier with language detection and translation support.
Supports 10+ languages with automatic language detection.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

LANGUAGE_FLAGS = {
    "en": "🇺🇸",
    "es": "🇪🇸",
    "fr": "🇫🇷",
    "de": "🇩🇪",
    "it": "🇮🇹",
    "pt": "🇵🇹",
    "ru": "🇷🇺",
    "zh": "🇨🇳",
    "ja": "🇯🇵",
    "ko": "🇰🇷",
}

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

# Language-specific models (pretrained sentiment models from Hugging Face)
LANGUAGE_MODELS = {
    "en": "distilbert-base-uncased-finetuned-sst-2-english",
    "es": "nlptown/bert-base-multilingual-uncased-sentiment",
    "fr": "nlptown/bert-base-multilingual-uncased-sentiment",
    "de": "nlptown/bert-base-multilingual-uncased-sentiment",
    "it": "nlptown/bert-base-multilingual-uncased-sentiment",
    "pt": "nlptown/bert-base-multilingual-uncased-sentiment",
    "ru": "nlptown/bert-base-multilingual-uncased-sentiment",
    "zh": "nlptown/bert-base-multilingual-uncased-sentiment",
    "ja": "nlptown/bert-base-multilingual-uncased-sentiment",
    "ko": "nlptown/bert-base-multilingual-uncased-sentiment",
}


@dataclass
class ClassificationResult:
    """Result of text classification."""

    text: str
    language: str
    language_flag: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str


class MultiLanguageClassifier:
    """
    Multi-language text classifier supporting 10+ languages.
    Automatically detects language and uses appropriate model.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize multi-language classifier.

        Args:
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.device = device
        self.pipelines = {}
        self.supported_languages = list(LANGUAGE_MODELS.keys())
        self.language_detector = None

        logger.info(f"Initializing Multi-Language Classifier on {device}")
        logger.info(
            f"Supported languages: {', '.join([LANGUAGE_FLAGS.get(lang, '🌐') + ' ' + LANGUAGE_NAMES.get(lang, lang) for lang in self.supported_languages])}"
        )

    def _get_language_detector(self):
        """Lazy load language detector."""
        if self.language_detector is None:
            try:
                import langdetect

                self.language_detector = langdetect
                logger.info("Language detector loaded: langdetect")
            except ImportError:
                logger.warning(
                    "langdetect not installed. Install with: pip install langdetect"
                )
                self.language_detector = None
        return self.language_detector

    def detect_language(self, text: str, default_lang: str = "en") -> str:
        """
        Detect text language.

        Args:
            text: Text to analyze
            default_lang: Default language if detection fails

        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        detector = self._get_language_detector()

        if detector is None:
            logger.warning(
                f"Language detection unavailable, using default: {default_lang}"
            )
            return default_lang

        try:
            from langdetect import detect, LangDetectException

            detected = detect(text)

            # Map langdetect codes to our codes if needed
            lang_mapping = {
                "pt-br": "pt",
                "pt-PT": "pt",
                "zh-cn": "zh",
                "zh-tw": "zh",
            }

            detected = lang_mapping.get(detected, detected)

            if detected in self.supported_languages:
                return detected
            else:
                logger.debug(
                    f"Detected language {detected} not in supported languages, using {default_lang}"
                )
                return default_lang
        except Exception as e:
            logger.debug(f"Language detection failed: {e}, using {default_lang}")
            return default_lang

    def _load_pipeline(self, language: str):
        """Load sentiment pipeline for a language."""
        if language in self.pipelines:
            return

        model_name = LANGUAGE_MODELS.get(language, LANGUAGE_MODELS["en"])

        try:
            logger.info(
                f"Loading model for {LANGUAGE_FLAGS.get(language, '🌐')} {LANGUAGE_NAMES.get(language, language)}: {model_name}"
            )

            self.pipelines[language] = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
            )

            logger.info(f"✅ Model loaded for {language}")
        except Exception as e:
            logger.error(f"Failed to load model for {language}: {e}")
            # Fallback to English
            if language != "en":
                logger.info("Falling back to English model")
                self._load_pipeline("en")
            else:
                raise

    def classify(
        self, text: str, language: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify text sentiment.

        Args:
            text: Text to classify
            language: Language code. If None, auto-detect

        Returns:
            ClassificationResult with predictions
        """
        # Detect language if not provided
        if language is None:
            language = self.detect_language(text)

        if language not in self.supported_languages:
            logger.warning(f"Language {language} not supported, using English")
            language = "en"

        # Load pipeline if needed
        self._load_pipeline(language)

        # Get predictions
        pipeline = self.pipelines[language]
        predictions = pipeline(text)

        # Parse predictions
        pred = predictions[0]
        label = pred["label"].lower()
        confidence = pred["score"]

        # Create probabilities dict
        probabilities = {
            label: confidence,
            "negative" if label == "positive" else "positive": 1.0 - confidence,
        }

        return ClassificationResult(
            text=text,
            language=language,
            language_flag=LANGUAGE_FLAGS.get(language, "🌐"),
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            model_used=LANGUAGE_MODELS.get(language, "unknown"),
        )

    def classify_batch(
        self, texts: List[str], languages: Optional[List[str]] = None
    ) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify
            languages: List of language codes. If None, auto-detect for each

        Returns:
            List of ClassificationResults
        """
        if languages is None:
            languages = [None] * len(texts)

        results = []
        for text, lang in zip(texts, languages):
            try:
                result = self.classify(text, lang)
                results.append(result)
            except Exception as e:
                logger.error(f"Error classifying text: {e}")
                results.append(None)

        return [r for r in results if r is not None]

    def get_language_distribution(self, texts: List[str]) -> Dict[str, int]:
        """
        Get distribution of languages in texts.

        Args:
            texts: List of texts

        Returns:
            Dictionary with language counts
        """
        distribution = {}
        for text in texts:
            lang = self.detect_language(text)
            distribution[lang] = distribution.get(lang, 0) + 1

        return distribution

    def summarize_results(self, results: List[ClassificationResult]) -> Dict:
        """
        Summarize classification results.

        Args:
            results: List of classification results

        Returns:
            Summary statistics
        """
        if not results:
            return {}

        total = len(results)
        by_language = {}
        by_label = {}

        total_confidence = 0

        for result in results:
            # By language
            if result.language not in by_language:
                by_language[result.language] = {"count": 0, "labels": {}}
            by_language[result.language]["count"] += 1
            by_language[result.language]["labels"][result.label] = (
                by_language[result.language]["labels"].get(result.label, 0) + 1
            )

            # By label
            by_label[result.label] = by_label.get(result.label, 0) + 1

            # Confidence
            total_confidence += result.confidence

        return {
            "total_samples": total,
            "by_language": by_language,
            "by_label": by_label,
            "average_confidence": total_confidence / total if total > 0 else 0,
            "supported_languages": self.supported_languages,
        }

    def export_results_json(self, results: List[ClassificationResult], filepath: str):
        """Export results to JSON."""
        import json

        data = {
            "timestamp": str(__import__("datetime").datetime.now()),
            "total_results": len(results),
            "results": [
                {
                    "text": r.text,
                    "language": r.language,
                    "label": r.label,
                    "confidence": float(r.confidence),
                    "probabilities": {k: float(v) for k, v in r.probabilities.items()},
                    "model_used": r.model_used,
                }
                for r in results
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results exported to {filepath}")
