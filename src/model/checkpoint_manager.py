"""
Checkpoint management for training and inference.
Handles saving, loading, and versioning of model checkpoints.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with versioning and cleanup.

    Features:
    - Save checkpoints with metadata
    - Load best or latest checkpoint
    - Automatic cleanup of old checkpoints
    - Resume training from checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 3,
        save_best_only: bool = False,
        metric_name: str = "loss",
        mode: str = "min"
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Only save when metric improves
            metric_name: Metric to track for best model
            mode: 'min' or 'max' for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.mode = mode

        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints: List[Dict[str, Any]] = []

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load existing checkpoint info
        self._load_checkpoint_info()

    def _load_checkpoint_info(self):
        """Load checkpoint information from disk."""
        info_file = self.checkpoint_dir / "checkpoint_info.json"
        if info_file.exists():
            try:
                with open(info_file, 'r') as f:
                    data = json.load(f)
                    self.checkpoints = data.get('checkpoints', [])
                    self.best_metric = data.get('best_metric', self.best_metric)
            except Exception as e:
                logger.warning(f"Could not load checkpoint info: {str(e)}")

    def _save_checkpoint_info(self):
        """Save checkpoint information to disk."""
        info_file = self.checkpoint_dir / "checkpoint_info.json"
        try:
            with open(info_file, 'w') as f:
                json.dump({
                    'checkpoints': self.checkpoints,
                    'best_metric': self.best_metric
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save checkpoint info: {str(e)}")

    def _is_better(self, metric: float) -> bool:
        """Check if metric is better than current best."""
        if self.mode == 'min':
            return metric < self.best_metric
        else:
            return metric > self.best_metric

    def should_save(self, metrics: Dict[str, float]) -> bool:
        """
        Determine if checkpoint should be saved based on metrics.

        Args:
            metrics: Dictionary of metric values

        Returns:
            True if checkpoint should be saved
        """
        if not self.save_best_only:
            return True

        if self.metric_name not in metrics:
            logger.warning(f"Metric '{self.metric_name}' not found in metrics")
            return True

        return self._is_better(metrics[self.metric_name])

    def save_checkpoint(
        self,
        model,
        tokenizer,
        metrics: Dict[str, float],
        step: int,
        extra_data: Optional[Dict] = None
    ) -> str:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            metrics: Current metrics
            step: Training step
            extra_data: Additional data to save

        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"checkpoint_{step}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Create checkpoint directory
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save model
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(checkpoint_path)
            else:
                logger.warning("Model doesn't have save_pretrained method")

            # Save tokenizer
            if tokenizer and hasattr(tokenizer, 'save_pretrained'):
                tokenizer.save_pretrained(checkpoint_path)

            # Save metadata
            metadata = {
                'step': step,
                'timestamp': timestamp,
                'metrics': metrics,
                'extra_data': extra_data or {}
            }

            with open(checkpoint_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            # Update best metric if applicable
            if self.metric_name in metrics and self._is_better(metrics[self.metric_name]):
                self.best_metric = metrics[self.metric_name]
                logger.info(f"New best {self.metric_name}: {self.best_metric:.4f}")

            # Record checkpoint
            self.checkpoints.append({
                'path': str(checkpoint_path),
                'step': step,
                'timestamp': timestamp,
                'metrics': metrics
            })

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            # Save checkpoint info
            self._save_checkpoint_info()

            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            # Cleanup failed checkpoint directory
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            raise

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            old_path = Path(oldest['path'])
            if old_path.exists():
                try:
                    shutil.rmtree(old_path)
                    logger.info(f"Removed old checkpoint: {old_path}")
                except Exception as e:
                    logger.warning(f"Could not remove old checkpoint: {str(e)}")

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        load_latest: bool = False
    ) -> Dict[str, Any]:
        """
        Load checkpoint information.

        Args:
            checkpoint_path: Specific checkpoint path to load
            load_best: Load the best checkpoint
            load_latest: Load the latest checkpoint

        Returns:
            Dictionary with checkpoint information
        """
        if checkpoint_path:
            checkpoint_dir = Path(checkpoint_path)
        elif load_best:
            # Find checkpoint with best metric
            if self.mode == 'min':
                checkpoint_dir = Path(min(self.checkpoints,
                                         key=lambda x: x['metrics'].get(self.metric_name, float('inf')))['path'])
            else:
                checkpoint_dir = Path(max(self.checkpoints,
                                         key=lambda x: x['metrics'].get(self.metric_name, float('-inf')))['path'])
        elif load_latest:
            checkpoint_dir = Path(self.checkpoints[-1]['path'])
        else:
            raise ValueError("Must specify checkpoint_path, load_best, or load_latest")

        # Load metadata
        metadata_path = checkpoint_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return {
            'path': str(checkpoint_dir),
            'metadata': metadata
        }

    def get_checkpoint_list(self) -> List[Dict[str, Any]]:
        """Get list of all checkpoints."""
        return self.checkpoints.copy()

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get information about the best checkpoint."""
        if not self.checkpoints:
            return None

        key = lambda x: x['metrics'].get(self.metric_name,
                                         float('inf') if self.mode == 'min' else float('-inf'))
        if self.mode == 'min':
            return min(self.checkpoints, key=key)
        else:
            return max(self.checkpoints, key=key)

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get information about the latest checkpoint."""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]

    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            True if successfully deleted
        """
        try:
            checkpoint_dir = Path(checkpoint_path)
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)

            # Remove from list
            self.checkpoints = [c for c in self.checkpoints if c['path'] != checkpoint_path]
            self._save_checkpoint_info()

            logger.info(f"Deleted checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {str(e)}")
            return False