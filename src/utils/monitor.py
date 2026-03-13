"""
Resource monitoring utilities for training and inference.
Tracks GPU, CPU, memory, and disk usage.
"""

import psutil
import torch
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from threading import Thread, Event
from queue import Queue
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResourceStats:
    """Container for resource statistics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_percent: float
    gpu_memory_allocated_gb: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        result = {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": self.memory_used_gb,
            "disk_percent": self.disk_percent,
        }
        if self.gpu_memory_allocated_gb is not None:
            result["gpu_memory_allocated_gb"] = self.gpu_memory_allocated_gb
        if self.gpu_memory_reserved_gb is not None:
            result["gpu_memory_reserved_gb"] = self.gpu_memory_reserved_gb
        if self.gpu_utilization is not None:
            result["gpu_utilization"] = self.gpu_utilization
        return result


class ResourceMonitor:
    """Monitor system resources during training."""

    def __init__(
        self,
        log_interval: float = 1.0,
        warning_threshold: Dict[str, float] = None
    ):
        """
        Initialize resource monitor.

        Args:
            log_interval: Interval in seconds between measurements
            warning_threshold: Thresholds for warnings (e.g., {"memory_percent": 90.0})
        """
        self.log_interval = log_interval
        self.warning_threshold = warning_threshold or {
            "memory_percent": 90.0,
            "gpu_memory_percent": 90.0,
            "disk_percent": 95.0
        }
        self._stop_event = Event()
        self._stats_queue: Queue = Queue()
        self._monitor_thread: Optional[Thread] = None

    @staticmethod
    def get_system_stats() -> Dict[str, float]:
        """Get current system resource usage."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "disk_percent": disk.percent
        }

    @staticmethod
    def get_gpu_stats() -> Dict[str, float]:
        """Get GPU statistics if available."""
        if not torch.cuda.is_available():
            return {}

        try:
            gpu_stats = {
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            }

            # Try to get GPU utilization if available
            if hasattr(torch.cuda, 'utilization'):
                gpu_stats["gpu_utilization"] = torch.cuda.utilization()

            return gpu_stats
        except Exception as e:
            logger.warning(f"Could not get GPU stats: {str(e)}")
            return {}

    def get_all_stats(self) -> ResourceStats:
        """Get combined system and GPU stats."""
        system_stats = self.get_system_stats()
        gpu_stats = self.get_gpu_stats()

        return ResourceStats(
            timestamp=time.time(),
            cpu_percent=system_stats["cpu_percent"],
            memory_percent=system_stats["memory_percent"],
            memory_used_gb=system_stats["memory_used_gb"],
            disk_percent=system_stats["disk_percent"],
            gpu_memory_allocated_gb=gpu_stats.get("gpu_memory_allocated_gb"),
            gpu_memory_reserved_gb=gpu_stats.get("gpu_memory_reserved_gb"),
            gpu_utilization=gpu_stats.get("gpu_utilization")
        )

    def check_warnings(self, stats: ResourceStats) -> List[str]:
        """Check for resource warnings based on thresholds."""
        warnings = []

        if stats.memory_percent > self.warning_threshold.get("memory_percent", 90):
            warnings.append(f"High memory usage: {stats.memory_percent:.1f}%")

        if stats.gpu_memory_allocated_gb is not None:
            gpu_percent = (stats.gpu_memory_allocated_gb /
                          self.warning_threshold.get("gpu_memory_total_gb", 24)) * 100
            if gpu_percent > self.warning_threshold.get("gpu_memory_percent", 90):
                warnings.append(f"High GPU memory usage: {gpu_percent:.1f}%")

        if stats.disk_percent > self.warning_threshold.get("disk_percent", 95):
            warnings.append(f"High disk usage: {stats.disk_percent:.1f}%")

        return warnings

    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            stats = self.get_all_stats()
            self._stats_queue.put(stats)

            # Check for warnings
            warnings = self.check_warnings(stats)
            for warning in warnings:
                logger.warning(warning)

            time.sleep(self.log_interval)

    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Resource monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Resource monitoring stopped")

    def get_latest_stats(self) -> Optional[ResourceStats]:
        """Get the most recent stats from queue."""
        stats = None
        while not self._stats_queue.empty():
            stats = self._stats_queue.get_nowait()
        return stats

    def get_all_recorded_stats(self) -> List[ResourceStats]:
        """Get all recorded stats from queue."""
        stats_list = []
        while not self._stats_queue.empty():
            stats_list.append(self._stats_queue.get_nowait())
        return stats_list


class TrainingMonitor:
    """
    Comprehensive training monitor that tracks resources and metrics.
    Provides callbacks for logging and early stopping.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        metrics_callback: Optional[Callable] = None,
        early_stopping_patience: int = 3,
        early_stopping_metric: str = "loss",
        early_stopping_mode: str = "min"
    ):
        """
        Initialize training monitor.

        Args:
            log_dir: Directory to save logs
            metrics_callback: Callback function for metrics logging
            early_stopping_patience: Number of steps to wait before early stopping
            early_stopping_metric: Metric to monitor for early stopping
            early_stopping_mode: 'min' or 'max' for early stopping
        """
        self.resource_monitor = ResourceMonitor()
        self.log_dir = log_dir
        self.metrics_callback = metrics_callback
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode

        self.metrics_history: List[Dict] = []
        self.best_metric = float('inf') if early_stopping_mode == 'min' else float('-inf')
        self.steps_without_improvement = 0
        self.should_stop = False

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics."""
        metrics_entry = {
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        self.metrics_history.append(metrics_entry)

        # Log resources
        stats = self.resource_monitor.get_all_stats()
        metrics_entry["resources"] = stats.to_dict()

        # Call callback if provided
        if self.metrics_callback:
            self.metrics_callback(metrics_entry)

        # Check early stopping
        self._check_early_stopping(metrics)

    def _check_early_stopping(self, metrics: Dict[str, float]):
        """Check if training should stop early."""
        if self.early_stopping_metric not in metrics:
            return

        current_value = metrics[self.early_stopping_metric]

        improved = False
        if self.early_stopping_mode == 'min':
            improved = current_value < self.best_metric
        else:
            improved = current_value > self.best_metric

        if improved:
            self.best_metric = current_value
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.early_stopping_patience:
            logger.info(
                f"Early stopping triggered: {self.early_stopping_metric} "
                f"hasn't improved for {self.steps_without_improvement} steps"
            )
            self.should_stop = True

    def start(self):
        """Start monitoring."""
        self.resource_monitor.start_monitoring()

    def stop(self):
        """Stop monitoring."""
        self.resource_monitor.stop_monitoring()

    def get_summary(self) -> Dict:
        """Get training summary."""
        if not self.metrics_history:
            return {}

        summary = {
            "total_steps": len(self.metrics_history),
            "best_metric": {
                "name": self.early_stopping_metric,
                "value": self.best_metric
            },
            "final_metrics": self.metrics_history[-1] if self.metrics_history else {}
        }

        # Calculate averages
        metric_names = [k for k in self.metrics_history[0].keys()
                       if k not in ['step', 'timestamp', 'resources']]
        for metric in metric_names:
            values = [m[metric] for m in self.metrics_history if metric in m]
            if values:
                summary[f"avg_{metric}"] = sum(values) / len(values)

        return summary


def get_system_summary() -> str:
    """Get human-readable system summary."""
    stats = ResourceMonitor.get_system_stats()
    gpu_stats = ResourceMonitor.get_gpu_stats()

    summary = [
        f"CPU Usage: {stats['cpu_percent']:.1f}%",
        f"Memory: {stats['memory_percent']:.1f}% ({stats['memory_used_gb']:.1f}GB used)",
        f"Disk: {stats['disk_percent']:.1f}%"
    ]

    if gpu_stats:
        summary.extend([
            f"GPU Memory Allocated: {gpu_stats.get('gpu_memory_allocated_gb', 0):.1f}GB",
            f"GPU Memory Reserved: {gpu_stats.get('gpu_memory_reserved_gb', 0):.1f}GB"
        ])
    else:
        summary.append("GPU: Not available")

    return " | ".join(summary)