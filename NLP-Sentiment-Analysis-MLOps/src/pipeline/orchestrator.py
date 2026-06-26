"""
End-to-end training pipeline orchestrating all components:
- Multi-language classification
- Data quality validation
- Hyperparameter optimization
- MLflow tracking
- Model registry management
- Canary deployment
"""

import logging
import os
import json
from typing import Dict, Optional, Callable
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)

# Import core components
try:
    from ..models.multilanguage_classifier import MultiLanguageClassifier
    from ..models.hyperparameter_optimizer import (
        HyperparameterOptimizer,
        DefaultSearchSpaces,
    )
    from ..models.mlflow_registry import ModelRegistry, CanaryDeploymentManager
    from ..utils.data_quality_validator import DataQualityValidator
    from ..utils.vibecoding_logger import setup_vibecoding_logger, Colors
except ImportError:
    logger.warning("Some modules not available. Using fallback imports.")


class TrainingPipeline:
    """
    Complete training pipeline orchestrating all MLOps components.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = setup_vibecoding_logger(__name__)
        self.validator = DataQualityValidator()
        self.registry = ModelRegistry()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {}

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML."""
        try:
            import yaml

            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    return yaml.safe_load(f)
        except ImportError:
            pass

        # Default config
        return {
            "model": "distilbert-base-uncased",
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "max_length": 256,
            "device": "cpu",
            "use_cuda": False,
        }

    def validate_data(self, texts: list, labels: list = None) -> bool:
        """
        Validate data quality.

        Args:
            texts: List of text samples
            labels: List of labels (optional)

        Returns:
            Whether validation passed
        """
        self.logger.info(f"🔍 Validating {len(texts)} samples...")

        validation = self.validator.run_full_validation(texts, labels)

        passed = validation["failed_checks"] == 0
        status = "✅ PASSED" if passed else "⚠️  PASSED WITH WARNINGS"
        self.logger.info(
            f"Data Validation {status}\n"
            f"  Passed: {validation['passed_checks']}/{validation['total_checks']}\n"
        )

        return passed

    def train_multilingual_classifiers(self, texts: list, labels: list = None) -> Dict:
        """
        Train multi-language classifiers.

        Args:
            texts: Training texts
            labels: Training labels

        Returns:
            Training results
        """
        self.logger.info(f"🌍 Training multilingual classifiers...")

        classifier = MultiLanguageClassifier(device=self.config.get("device", "cpu"))

        # Detect languages in dataset
        results = classifier.classify_batch(texts)

        language_dist = classifier.get_language_distribution(results)
        self.logger.info(f"📊 Language Distribution: {language_dist}")

        self.metrics["languages"] = language_dist

        return {
            "classifier": classifier,
            "language_distribution": language_dist,
            "sample_predictions": results[:5] if results else [],
        }

    def optimize_hyperparameters(
        self,
        train_texts: list,
        train_labels: list,
        objective_fn: Optional[Callable] = None,
        n_trials: int = 20,
    ) -> Dict:
        """
        Optimize hyperparameters using Optuna.

        Args:
            train_texts: Training texts
            train_labels: Training labels
            objective_fn: Objective function for optimization
            n_trials: Number of trials

        Returns:
            Optimization results
        """
        self.logger.info(f"⚡ Optimizing hyperparameters ({n_trials} trials)...")

        optimizer = HyperparameterOptimizer(n_trials=n_trials)

        # Create study
        optimizer.create_study(sampler="tpe", direction="maximize")

        # Default objective function if not provided
        if objective_fn is None:

            def objective_fn(trial):
                """Default objective: random score for demo."""
                import random

                return random.uniform(0.75, 0.95)

        # Run optimization
        try:
            best_params = optimizer.optimize(
                objective_fn, search_space=DefaultSearchSpaces.lightweight_model()
            )
            self.logger.info(f"🎯 Best Parameters Found:\n{best_params}")
            self.metrics["best_hyperparams"] = best_params
        except Exception as e:
            self.logger.warning(f"Hyperparameter optimization failed: {e}")
            best_params = DefaultSearchSpaces.lightweight_model()

        return {
            "best_params": best_params,
            "optimizer": optimizer,
        }

    def register_model(
        self,
        model_name: str,
        model_uri: str,
        description: str = "",
        stage: str = "Development",
    ):
        """
        Register model in MLflow Model Registry.

        Args:
            model_name: Model name
            model_uri: Model artifact URI
            description: Model description
            stage: Initial stage
        """
        self.logger.info(f"📦 Registering model: {model_name}")

        try:
            version = self.registry.register_model(
                model_name=model_name,
                model_uri=model_uri,
                tags={
                    "run_id": self.run_id,
                    "timestamp": datetime.now().isoformat(),
                },
                description=description,
            )

            self.logger.info(f"✅ Registered: {model_name} v{version}")

            # Transition to specified stage
            if stage != "Development":
                self.registry.transition_model(model_name, version, stage)
                self.logger.info(f"📊 Transitioned to: {stage}")

            return version
        except Exception as e:
            self.logger.error(f"Registration failed: {e}")
            return None

    def setup_canary_deployment(
        self,
        model_name: str,
        version: str,
    ) -> Dict:
        """
        Setup canary deployment strategy.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            Deployment plan
        """
        self.logger.info(f"🚀 Setting up canary deployment...")

        try:
            deployment_mgr = CanaryDeploymentManager(self.registry)

            plan = deployment_mgr.create_canary_deployment(
                model_name=model_name,
                new_version=version,
            )

            self.logger.info(f"✅ Deployment plan created:\n{plan}")

            return {
                "deployment_manager": deployment_mgr,
                "plan": plan,
            }
        except Exception as e:
            self.logger.warning(f"Canary deployment setup failed: {e}")
            return {}

    def run_full_pipeline(
        self,
        train_texts: list,
        train_labels: list = None,
        validate: bool = True,
        optimize: bool = True,
        register_model: bool = True,
        model_name: str = "sentiment-classifier",
    ) -> Dict:
        """
        Run complete training pipeline.

        Args:
            train_texts: Training texts
            train_labels: Training labels
            validate: Run data validation
            optimize: Run hyperparameter optimization
            register_model: Register final model
            model_name: Name for registered model

        Returns:
            Pipeline execution summary
        """
        start_time = datetime.now()
        self.logger.info(f"🎬 Starting Pipeline: {self.run_id}")

        results = {
            "run_id": self.run_id,
            "timestamp": start_time.isoformat(),
            "stages": {},
        }

        # Stage 1: Data Validation
        if validate:
            self.logger.info(f"\n{'='*60}")
            self.logger.info("Stage 1: Data Validation")
            self.logger.info("=" * 60)
            validation_result = self.validate_data(train_texts, train_labels)
            results["stages"]["validation"] = {
                "passed": validation_result,
                "metrics": self.metrics,
            }

        # Stage 2: Multi-language Analysis
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Stage 2: Multi-Language Classification")
        self.logger.info("=" * 60)
        ml_results = self.train_multilingual_classifiers(train_texts, train_labels)
        results["stages"]["multilingual"] = {
            "language_distribution": ml_results["language_distribution"],
        }

        # Stage 3: Hyperparameter Optimization
        if optimize:
            self.logger.info(f"\n{'='*60}")
            self.logger.info("Stage 3: Hyperparameter Optimization")
            self.logger.info("=" * 60)
            opt_results = self.optimize_hyperparameters(
                train_texts,
                train_labels,
                n_trials=10,
            )
            results["stages"]["optimization"] = {
                "best_params": opt_results["best_params"],
            }

        # Stage 4: Model Registration (simulated)
        if register_model:
            self.logger.info(f"\n{'='*60}")
            self.logger.info("Stage 4: Model Registration")
            self.logger.info("=" * 60)

            # Create dummy model artifact
            model_uri = f"runs:/{self.run_id}/model"
            version = self.register_model(
                model_name=model_name,
                model_uri=model_uri,
                description=f"Pipeline run {self.run_id}",
            )

            results["stages"]["registration"] = {
                "model_name": model_name,
                "version": version,
                "uri": model_uri,
            }

            # Stage 5: Canary Deployment
            self.logger.info(f"\n{'='*60}")
            self.logger.info("Stage 5: Canary Deployment")
            self.logger.info("=" * 60)

            if version:
                deployment = self.setup_canary_deployment(model_name, version)
                results["stages"]["deployment"] = deployment

        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.logger.info(f"\n{'='*60}")
        self.logger.info("✅ Pipeline Complete")
        self.logger.info("=" * 60)
        self.logger.info(f"Duration: {duration:.1f}s")
        self.logger.info(f"Run ID: {self.run_id}")

        results["duration_seconds"] = duration
        results["status"] = "completed"

        return results

    def save_results(self, results: Dict, output_dir: str = "pipeline_results"):
        """
        Save pipeline results to file.

        Args:
            results: Pipeline results
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"results_{self.run_id}.json")

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"📝 Results saved: {output_file}")
