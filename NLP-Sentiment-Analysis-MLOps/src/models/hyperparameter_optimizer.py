"""
Automated hyperparameter optimization using Optuna.
Finds optimal parameters for model training with visualization.
"""

from __future__ import annotations

import logging
from typing import Dict, Callable, Optional, Tuple, TYPE_CHECKING
import json
from pathlib import Path

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Install with: pip install optuna")


class HyperparameterOptimizer:
    """
    Automated hyperparameter tuning using Optuna.
    Performs Bayesian optimization to find optimal training parameters.
    """

    def __init__(self, n_trials: int = 100, n_jobs: int = 1, verbose: bool = True):
        """
        Initialize optimizer.

        Args:
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            verbose: Print progress
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.study = None
        self.best_params = None
        self.trials_history = []

    def create_study(
        self,
        study_name: str = "sentiment-optimization",
        direction: str = "maximize",
        sampler: str = "tpe",
    ):
        """
        Create Optuna study.

        Args:
            study_name: Name of the study
            direction: 'maximize' or 'minimize'
            sampler: Sampler type ('tpe', 'random', 'grid')

        Returns:
            optuna.Study object
        """
        if sampler == "tpe":
            sampler_obj = optuna.samplers.TPESampler(seed=42)
        elif sampler == "random":
            sampler_obj = optuna.samplers.RandomSampler(seed=42)
        else:
            sampler_obj = optuna.samplers.TPESampler(seed=42)

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler_obj,
        )

        logger.info(f"Created study: {study_name} (direction: {direction})")
        return self.study

    def suggest_hyperparameters(self, trial: "optuna.Trial", search_space: Dict) -> Dict:
        """
        Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial object
            search_space: Dictionary defining search space
                Example: {
                    'learning_rate': ('float', 1e-5, 1e-3, 'log'),
                    'batch_size': ('int', 8, 64, 1),
                    'num_epochs': ('int', 1, 5, 1),
                    'dropout': ('float', 0.1, 0.5),
                }

        Returns:
            Dictionary of suggested parameters
        """
        params = {}

        for param_name, param_config in search_space.items():
            param_type = param_config[0]

            if param_type == "float":
                low, high = param_config[1], param_config[2]
                log = param_config[3] == "log" if len(param_config) > 3 else False
                params[param_name] = trial.suggest_float(param_name, low, high, log=log)

            elif param_type == "int":
                low, high = param_config[1], param_config[2]
                params[param_name] = trial.suggest_int(param_name, low, high)

            elif param_type == "categorical":
                choices = param_config[1]
                params[param_name] = trial.suggest_categorical(param_name, choices)

        return params

    def optimize(
        self,
        objective: Callable,
        search_space: Dict,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Dict:
        """
        Run hyperparameter optimization.

        Args:
            objective: Objective function that takes (trial, search_space) and returns float
            search_space: Dictionary defining search space
            n_trials: Number of trials (overrides init value)
            timeout: Timeout in seconds

        Returns:
            Best parameters found
        """
        if self.study is None:
            self.create_study()

        n_trials = n_trials or self.n_trials

        def wrapped_objective(trial):
            params = self.suggest_hyperparameters(trial, search_space)
            score = objective(trial, params)

            self.trials_history.append(
                {
                    "trial": trial.number,
                    "params": params,
                    "score": score,
                }
            )

            if self.verbose and trial.number % 10 == 0:
                logger.info(f"Trial {trial.number}: Score = {score:.4f}")

            return score

        logger.info(f"Starting optimization with {n_trials} trials...")
        self.study.optimize(
            wrapped_objective,
            n_trials=n_trials,
            n_jobs=self.n_jobs,
            timeout=timeout,
            show_progress_bar=self.verbose,
        )

        self.best_params = self.study.best_params
        logger.info(f"Optimization completed. Best score: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_params

    def get_best_params(self) -> Dict:
        """Get best parameters found."""
        if self.best_params is None and self.study is not None:
            self.best_params = self.study.best_params
        return self.best_params or {}

    def get_trials_dataframe(self):
        """Get trials as DataFrame."""
        try:
            import pandas as pd

            return self.study.trials_dataframe()
        except ImportError:
            logger.warning("pandas not available")
            return None

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.

        Args:
            save_path: Path to save plot (optional)
        """
        if self.study is None:
            logger.warning("No study to plot")
            return

        try:
            fig = plot_optimization_history(self.study)

            if save_path:
                fig.write_html(save_path)
                logger.info(f"Optimization history plot saved to {save_path}")
            else:
                fig.show()
        except Exception as e:
            logger.error(f"Failed to plot optimization history: {e}")

    def plot_param_importances(self, save_path: Optional[str] = None):
        """
        Plot parameter importances.

        Args:
            save_path: Path to save plot (optional)
        """
        if self.study is None:
            logger.warning("No study to plot")
            return

        try:
            fig = plot_param_importances(self.study)

            if save_path:
                fig.write_html(save_path)
                logger.info(f"Parameter importances plot saved to {save_path}")
            else:
                fig.show()
        except Exception as e:
            logger.error(f"Failed to plot parameter importances: {e}")

    def save_study(self, filepath: str):
        """Save study to file."""
        if self.study is None:
            logger.warning("No study to save")
            return

        # Save using SQLite backend
        import sqlite3

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        try:
            self.study.save_to_sqlite(filepath)
            logger.info(f"Study saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save study: {e}")

    def load_study(self, filepath: str, study_name: str = "sentiment-optimization"):
        """Load study from file."""
        try:
            self.study = optuna.load_study(
                study_name=study_name, storage=f"sqlite:///{filepath}"
            )
            self.best_params = self.study.best_params
            logger.info(f"Study loaded from {filepath}")
            return self.study
        except Exception as e:
            logger.error(f"Failed to load study: {e}")
            return None

    def print_summary(self):
        """Print optimization summary."""
        if self.study is None:
            logger.warning("No study to summarize")
            return

        print("\n" + "=" * 60)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"\nBest Value: {self.study.best_value:.6f}")
        print(f"Best Trial: {self.study.best_trial.number}")
        print("\nBest Parameters:")
        for key, value in self.best_params.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.6f}")
            else:
                print(f"  {key:20s}: {value}")

        print(f"\nTotal Trials: {len(self.study.trials)}")
        print(
            f"Completed Trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
        )
        print("=" * 60 + "\n")


class DefaultSearchSpaces:
    """Default search spaces for common parameters."""

    @staticmethod
    def bert_training() -> Dict:
        """Default search space for BERT training."""
        return {
            "learning_rate": ("float", 1e-5, 5e-5, "log"),
            "batch_size": ("int", 16, 64, 1),
            "num_epochs": ("int", 2, 5, 1),
            "warmup_steps": ("int", 100, 500, 1),
            "weight_decay": ("float", 0.0, 0.1),
            "dropout": ("float", 0.1, 0.3),
        }

    @staticmethod
    def lightweight_model() -> Dict:
        """Search space for lightweight models (DistilBERT, etc)."""
        return {
            "learning_rate": ("float", 2e-5, 1e-4, "log"),
            "batch_size": ("int", 8, 32, 1),
            "num_epochs": ("int", 1, 3, 1),
            "warmup_steps": ("int", 50, 200, 1),
        }

    @staticmethod
    def ensemble_params() -> Dict:
        """Search space for ensemble methods."""
        return {
            "n_models": ("int", 3, 10, 1),
            "voting_type": ("categorical", ["hard", "soft"]),
            "model_diversity": ("float", 0.1, 0.9),
        }
