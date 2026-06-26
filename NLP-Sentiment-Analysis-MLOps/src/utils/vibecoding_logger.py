"""
Vibecoding-styled logger with rainbow colors, emojis, and ASCII art.
"""

import logging
import sys
from typing import Optional
from datetime import datetime


# ANSI Color Codes
class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    ORANGE = "\033[38;5;208m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


class VibecodeFormatter(logging.Formatter):
    """Custom formatter with vibecoding style - colored logs with emojis."""

    COLORS_MAP = {
        "DEBUG": Colors.CYAN,
        "INFO": Colors.GREEN,
        "WARNING": Colors.ORANGE,
        "ERROR": Colors.RED,
        "CRITICAL": Colors.RED + Colors.BOLD,
    }

    EMOJI_MAP = {
        "DEBUG": "🔍",
        "INFO": "✨",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🚨",
    }

    def format(self, record):
        """Format log record with colors and emojis."""
        level_name = record.levelname
        color = self.COLORS_MAP.get(level_name, Colors.WHITE)
        emoji = self.EMOJI_MAP.get(level_name, "•")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format: emoji [LEVEL] timestamp | message
        formatted = (
            f"{emoji} {color}[{level_name:8s}]{Colors.END} "
            f"{Colors.BLUE}{timestamp}{Colors.END} | "
            f"{record.getMessage()}"
        )

        return formatted


def setup_vibecoding_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Setup a vibecoding-styled logger.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(VibecodeFormatter())

    # Remove existing handlers
    logger.handlers = []
    logger.addHandler(console_handler)

    return logger


class PipelineVisualizer:
    """ASCII art visualizations for ML pipelines."""

    @staticmethod
    def print_training_start(model_name: str, language: str = "🌐"):
        """Print training start banner."""
        banner = f"""
╔════════════════════════════════════════════════════════════╗
║  {language} STARTING MULTI-LANGUAGE TEXT CLASSIFIER TRAINING
║  Model: {model_name:43s} ║
╚════════════════════════════════════════════════════════════╝
        """
        print(Colors.BOLD + Colors.CYAN + banner + Colors.END)

    @staticmethod
    def print_pipeline_stage(stage: int, total: int, stage_name: str, emoji: str = "→"):
        """Print pipeline progress."""
        progress = "▓" * stage + "░" * (total - stage)
        print(
            f"\n{Colors.BLUE}{emoji} Stage {stage}/{total}: "
            f"{Colors.BOLD}{stage_name}{Colors.END}"
            f"\n  [{progress}]"
        )

    @staticmethod
    def print_language_status(languages: dict):
        """Print language-wise status with emojis."""
        flags = {
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

        print(f"\n{Colors.BOLD}Language Support:{Colors.END}")
        for lang, status in languages.items():
            flag = flags.get(lang, "🌐")
            status_icon = "✅" if status else "⏳"
            print(f"  {flag} {lang.upper():6s} {status_icon}")

    @staticmethod
    def print_confusion_matrix_heatmap(matrix: list, labels: list):
        """Print confusion matrix as terminal heatmap."""
        import numpy as np

        # Normalize for visualization
        matrix_norm = np.array(matrix) / (np.array(matrix).max() + 1e-8)

        # Heat map colors
        heat_levels = "░▒▓█"

        print(f"\n{Colors.BOLD}Confusion Matrix Heatmap:{Colors.END}")
        print(f"  {' '.join(f'{l:6s}' for l in labels)}")

        for i, row in enumerate(matrix_norm):
            heat_str = ""
            for val in row:
                idx = min(int(val * 3), 3)
                heat_str += heat_levels[idx] * 2 + " "
            print(f"{labels[i]:6s} {heat_str}")

    @staticmethod
    def print_metrics_comparison(metrics: dict, emoji: str = "📊"):
        """Print metrics in a nice box."""
        print(f"\n{emoji} {Colors.BOLD}Performance Metrics:{Colors.END}")

        for key, value in metrics.items():
            if isinstance(value, float):
                bar_length = int(value * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {key:20s} {Colors.GREEN}[{bar}]{Colors.END} {value:.4f}")
            else:
                print(f"  {key:20s} {Colors.CYAN}{value}{Colors.END}")


class LoadingAnimation:
    """Animated loading sequences."""

    SPINNERS = {
        "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "line": ["-", "\\", "|", "/"],
        "arrow": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
        "bar": ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"],
    }

    def __init__(self, text: str = "Processing", spinner_type: str = "dots"):
        self.text = text
        self.spinner = self.SPINNERS.get(spinner_type, self.SPINNERS["dots"])
        self.index = 0

    def next_frame(self) -> str:
        """Get next animation frame."""
        frame = self.spinner[self.index % len(self.spinner)]
        self.index += 1
        return f"\r{Colors.CYAN}{frame}{Colors.END} {self.text}... "

    def reset(self):
        """Reset animation."""
        self.index = 0


class RainbowProgressBar:
    """Rainbow-colored progress bar."""

    COLORS = [
        Colors.RED,
        Colors.ORANGE,
        Colors.YELLOW,
        Colors.GREEN,
        Colors.BLUE,
        Colors.PURPLE,
    ]

    @staticmethod
    def print_progress(current: int, total: int, prefix: str = "", emoji: str = ""):
        """Print rainbow progress bar."""
        percent = current / total
        filled = int(percent * 30)

        # Rainbow coloring
        color_idx = int(percent * len(RainbowProgressBar.COLORS))
        color = RainbowProgressBar.COLORS[
            min(color_idx, len(RainbowProgressBar.COLORS) - 1)
        ]

        bar = "█" * filled + "░" * (30 - filled)
        print(
            f"\r{emoji} {prefix:20s} {color}[{bar}]{Colors.END} "
            f"{percent*100:5.1f}% ({current}/{total})",
            end="",
            flush=True,
        )

        if current == total:
            print()  # New line at completion


def print_deployment_strategy():
    """Print canary deployment strategy ASCII diagram."""
    diagram = f"""
{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════╗
║        CANARY DEPLOYMENT STRATEGY                         ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  Stage 1 (5%)   ████░░░░░░░░░░░░░░░  → 5% Traffic        ║
║  Stage 2 (25%)  ████████████░░░░░░░░  → 25% Traffic       ║
║  Stage 3 (50%)  ████████████████░░░░  → 50% Traffic       ║
║  Stage 4 (100%) ████████████████████  → 100% Traffic (Prod)║
║                                                           ║
║  Rollback Point: ❌ If error rate > threshold at any stage║
║  Health Check: 🏥 Monitor metrics continuously            ║
║  Version Control: 🔄 Keep previous model for safety       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝{Colors.END}
    """
    print(diagram)


def print_mlflow_registry_status():
    """Print MLflow Model Registry status."""
    status = f"""
{Colors.BOLD}{Colors.GREEN}╔═══════════════════════════════════════════════════════════╗
║        MLflow Model Registry                             ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  Development    🔄 Active training and experimentation    ║
║  ↓                                                        ║
║  Staging        🧪 Pre-production validation              ║
║  ↓                                                        ║
║  Production     ✅ Live model serving to users            ║
║                                                           ║
║  Promotion Flow:                                          ║
║  New Model → Dev → Staging → Prod (with approval)        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝{Colors.END}
    """
    print(status)
