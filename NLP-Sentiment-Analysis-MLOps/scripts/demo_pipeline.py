"""
Complete end-to-end demo showing all components working together.
Run this to see the full pipeline in action.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.orchestrator import TrainingPipeline
from src.models.multilanguage_classifier import MultiLanguageClassifier
from src.models.hyperparameter_optimizer import (
    HyperparameterOptimizer,
    DefaultSearchSpaces,
)
from src.utils.vibecoding_logger import (
    setup_vibecoding_logger,
    PipelineVisualizer,
    LoadingAnimation,
)
from src.utils.data_quality_validator import DataQualityValidator

# Setup logging
logger = setup_vibecoding_logger(__name__)


def demo_data_quality():
    """Demonstrate data quality validation."""
    logger.info("\n" + "=" * 70)
    logger.info("🔍 DATA QUALITY VALIDATION DEMO")
    logger.info("=" * 70)

    validator = DataQualityValidator()

    # Sample texts
    texts = [
        "I absolutely love this product! It's amazing!",
        "This is terrible and I hate it.",
        "The quality is decent but could be better.",
        "Best purchase ever!",
        "Would not recommend.",
        "Excellent customer service and fast delivery.",
        "Poor quality and not worth the price.",
        "Amazing value for money!",
    ]

    labels = [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
    ]

    # Run validation
    results = validator.run_full_validation(texts, labels)

    # Print report
    validator.print_validation_report()

    return results


def demo_multilanguage():
    """Demonstrate multi-language classification."""
    logger.info("\n" + "=" * 70)
    logger.info("🌍 MULTI-LANGUAGE CLASSIFICATION DEMO")
    logger.info("=" * 70)

    classifier = MultiLanguageClassifier(device="cpu")

    # Multi-language texts
    multilang_texts = [
        "I love this product!",  # English
        "¡Me encanta este producto!",  # Spanish
        "J'aime ce produit!",  # French
        "Ich liebe dieses Produkt!",  # German
        "Amo questo prodotto!",  # Italian
        "Adorei este produto!",  # Portuguese
    ]

    logger.info(f"Classifying {len(multilang_texts)} texts in different languages...")

    results = classifier.classify_batch(multilang_texts)

    logger.info(f"\n📊 Results:")
    for result in results:
        emoji = result.language_flag
        lang = result.language or "unknown"
        label = result.label or "N/A"
        confidence = result.confidence or 0.0
        logger.info(
            f"  {emoji} {lang:10s} | {result.text[:40]:40s} | {label:10s} ({confidence:.2%})"
        )

    return results


def demo_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization."""
    logger.info("\n" + "=" * 70)
    logger.info("⚡ HYPERPARAMETER OPTIMIZATION DEMO")
    logger.info("=" * 70)

    optimizer = HyperparameterOptimizer(n_trials=10)

    logger.info("Creating optimization study with TPE sampler...")
    optimizer.create_study(sampler="tpe", direction="maximize")

    # Simple objective function for demo
    def objective(trial):
        """Simple objective function."""
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return -(x**2 + y**2)  # Maximize = minimize negative

    logger.info(f"Running {optimizer.n_trials} optimization trials...")

    best_params = optimizer.optimize(objective, search_space=None)

    logger.info(f"\n✅ Best Parameters: {best_params}")

    return optimizer


def demo_full_pipeline():
    """Demonstrate complete training pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("🎬 FULL TRAINING PIPELINE DEMO")
    logger.info("=" * 70)

    # Sample training data
    train_texts = [
        "I absolutely love this product!",
        "This is terrible and I hate it.",
        "The quality is decent but could be better.",
        "Best purchase ever!",
        "Would not recommend.",
        "Excellent customer service and fast delivery.",
        "Poor quality and not worth the price.",
        "Amazing value for money!",
        "I'm very satisfied with my purchase.",
        "Not worth the money.",
    ]

    train_labels = [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "positive",
        "negative",
    ]

    # Initialize pipeline
    pipeline = TrainingPipeline()

    # Run full pipeline
    results = pipeline.run_full_pipeline(
        train_texts=train_texts,
        train_labels=train_labels,
        validate=True,
        optimize=True,
        register_model=True,
        model_name="multilingual-sentiment",
    )

    # Save results
    pipeline.save_results(results)

    return results


def demo_visualizations():
    """Demonstrate visualization components."""
    logger.info("\n" + "=" * 70)
    logger.info("🎨 VISUALIZATION COMPONENTS DEMO")
    logger.info("=" * 70)

    visualizer = PipelineVisualizer()

    logger.info("\n1️⃣  Training Pipeline Start:")
    visualizer.print_training_start()

    logger.info("\n2️⃣  Pipeline Stages:")
    stages = [
        "Data Loading",
        "Data Validation",
        "Preprocessing",
        "Training",
        "Evaluation",
    ]
    for i, stage in enumerate(stages, 1):
        visualizer.print_pipeline_stage(stage, i, len(stages))

    logger.info("\n3️⃣  Language Support Status:")
    language_metrics = {
        "English": {"samples": 1200, "accuracy": 0.94},
        "Spanish": {"samples": 950, "accuracy": 0.91},
        "French": {"samples": 850, "accuracy": 0.89},
    }
    visualizer.print_language_status(language_metrics)

    logger.info("\n4️⃣  Loading Animations:")
    animation = LoadingAnimation()

    logger.info("  Dots animation:")
    animation.animate(animation.dots, duration=2)

    logger.info("\n  Bar animation:")
    animation.animate(animation.bar, duration=2)


def main():
    """Run all demos."""
    logger.info("\n" + "#" * 70)
    logger.info("# MULTILINGUAL TEXT CLASSIFIER - COMPLETE SYSTEM DEMO")
    logger.info("#" * 70)

    try:
        # Run demos
        demo_data_quality()
        demo_multilanguage()
        demo_hyperparameter_optimization()
        demo_visualizations()
        demo_full_pipeline()

        logger.info("\n" + "#" * 70)
        logger.info("# ✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        logger.info("#" * 70)

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
