"""
Industrial-grade logging configuration for the fine-tuning system.
Provides structured logging with file rotation and console output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    name: str = "industrial_lora",
    log_dir: str = "./logs",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup a logger with both file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory to store log files
        level: Logging level
        max_bytes: Maximum size of each log file
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler with rotation
    log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class TrainingLogger:
    """
    Specialized logger for training progress tracking.
    Logs metrics, checkpoints, and training progress.
    """

    def __init__(self, log_dir: str = "./logs"):
        self.logger = setup_logger("training", log_dir)
        self.metrics_logger = setup_logger("metrics", log_dir)
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float('inf')

    def log_training_start(self, config: dict):
        """Log training configuration at start."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING SESSION STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration: {config}")

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log start of training epoch."""
        self.current_epoch = epoch
        self.logger.info(f"Starting Epoch {epoch}/{total_epochs}")

    def log_step(self, step: int, loss: float, learning_rate: float):
        """Log training step progress."""
        self.current_step = step
        self.metrics_logger.info(f"Step {step}: loss={loss:.4f}, lr={learning_rate:.2e}")

        if step % 100 == 0:
            self.logger.info(f"Step {step}: loss={loss:.4f}")

    def log_checkpoint(self, checkpoint_path: str, is_best: bool = False):
        """Log checkpoint save."""
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        if is_best:
            self.logger.info(f"New best model! Checkpoint: {checkpoint_path}")

    def log_training_end(self, final_metrics: dict):
        """Log training completion."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING SESSION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Final Metrics: {final_metrics}")

    def log_error(self, error: Exception, context: str = ""):
        """Log error with context."""
        self.logger.error(f"Error during {context}: {str(error)}")
        self.logger.exception(error)


def get_logger(name: str = "industrial_lora") -> logging.Logger:
    """Get or create a logger by name."""
    return logging.getLogger(name)