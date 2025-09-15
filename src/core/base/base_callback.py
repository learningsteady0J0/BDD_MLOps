"""Base callback class for vision training hooks."""

from typing import Any, Dict, Optional, List
from pathlib import Path
import json
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf


class BaseVisionCallback(Callback):
    """
    Base callback class for vision-specific training hooks.

    Provides common functionality for:
    - Metrics tracking and visualization
    - Image logging and visualization
    - Model checkpointing strategies
    - Training progress monitoring
    """

    def __init__(
        self,
        config: Optional[DictConfig] = None,
        log_dir: Optional[Path] = None,
        verbose: bool = True
    ):
        """
        Initialize the base callback.

        Args:
            config: Configuration object
            log_dir: Directory for logging outputs
            verbose: Whether to print verbose messages
        """
        super().__init__()
        self.config = config
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Tracking variables
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": {},
            "val_metrics": {},
            "learning_rates": [],
            "epochs": []
        }
        self.start_time = None
        self.epoch_start_time = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training starts."""
        self.start_time = time.time()
        if self.verbose:
            print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total epochs: {trainer.max_epochs}")
            print(f"Device: {pl_module.device}")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the start of each training epoch."""
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of each training epoch."""
        # Record metrics
        metrics = trainer.callback_metrics
        self.training_history["epochs"].append(trainer.current_epoch)

        if "train/loss" in metrics:
            self.training_history["train_loss"].append(metrics["train/loss"].item())

        # Record learning rate
        lr = self._get_learning_rate(pl_module)
        self.training_history["learning_rates"].append(lr)

        # Time tracking
        epoch_time = time.time() - self.epoch_start_time
        if self.verbose:
            print(f"Epoch {trainer.current_epoch} completed in {epoch_time:.2f}s")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of each validation epoch."""
        metrics = trainer.callback_metrics

        if "val/loss" in metrics:
            self.training_history["val_loss"].append(metrics["val/loss"].item())

        # Store other validation metrics
        for key, value in metrics.items():
            if key.startswith("val/") and key != "val/loss":
                metric_name = key.replace("val/", "")
                if metric_name not in self.training_history["val_metrics"]:
                    self.training_history["val_metrics"][metric_name] = []
                self.training_history["val_metrics"][metric_name].append(value.item())

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training ends."""
        total_time = time.time() - self.start_time
        if self.verbose:
            print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")

        # Save training history
        self._save_training_history()

        # Generate final plots
        self._plot_training_history()

    def _get_learning_rate(self, pl_module: pl.LightningModule) -> float:
        """Get current learning rate from optimizer."""
        optimizer = pl_module.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        for param_group in optimizer.param_groups:
            return param_group["lr"]
        return 0.0

    def _save_training_history(self) -> None:
        """Save training history to JSON file."""
        history_file = self.log_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(self.training_history, f, indent=2, default=str)

    def _plot_training_history(self) -> None:
        """Generate and save training history plots."""
        # This method can be overridden in subclasses for custom plotting
        pass


class VisualizationCallback(BaseVisionCallback):
    """
    Callback for visualizing model predictions and activations.
    """

    def __init__(
        self,
        config: Optional[DictConfig] = None,
        log_dir: Optional[Path] = None,
        log_every_n_epochs: int = 5,
        num_samples: int = 8,
        **kwargs
    ):
        """
        Initialize visualization callback.

        Args:
            config: Configuration object
            log_dir: Directory for saving visualizations
            log_every_n_epochs: Frequency of logging visualizations
            num_samples: Number of samples to visualize
            **kwargs: Additional arguments for base class
        """
        super().__init__(config, log_dir, **kwargs)
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        """Log visualizations at the end of validation batches."""
        # Only log on first batch every n epochs
        if batch_idx == 0 and trainer.current_epoch % self.log_every_n_epochs == 0:
            self._visualize_predictions(pl_module, batch)

    def _visualize_predictions(self, pl_module: pl.LightningModule, batch: tuple) -> None:
        """Visualize model predictions on a batch."""
        images, labels = batch
        images = images[:self.num_samples]
        labels = labels[:self.num_samples]

        # Get predictions
        with torch.no_grad():
            predictions = pl_module(images)
            if predictions.dim() > 1:
                predictions = predictions.argmax(dim=1)

        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()

        for i, (img, label, pred) in enumerate(zip(images, labels, predictions)):
            # Denormalize image
            img = self._denormalize_image(img, pl_module)

            axes[i].imshow(img)
            axes[i].set_title(f"True: {label.item()}, Pred: {pred.item()}")
            axes[i].axis("off")

        plt.tight_layout()

        # Save figure
        save_path = self.log_dir / f"predictions_epoch_{pl_module.current_epoch}.png"
        plt.savefig(save_path)
        plt.close()

    def _denormalize_image(self, img: torch.Tensor, pl_module: pl.LightningModule) -> np.ndarray:
        """Denormalize and convert image for visualization."""
        img = img.cpu()

        # Default ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Denormalize
        for i in range(3):
            img[i] = img[i] * std[i] + mean[i]

        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()

        return img


class ModelAnalysisCallback(BaseVisionCallback):
    """
    Callback for analyzing model behavior during training.
    """

    def __init__(
        self,
        config: Optional[DictConfig] = None,
        log_dir: Optional[Path] = None,
        analyze_gradients: bool = True,
        analyze_weights: bool = True,
        analyze_activations: bool = False,
        **kwargs
    ):
        """
        Initialize model analysis callback.

        Args:
            config: Configuration object
            log_dir: Directory for saving analysis
            analyze_gradients: Whether to analyze gradients
            analyze_weights: Whether to analyze weights
            analyze_activations: Whether to analyze activations
            **kwargs: Additional arguments for base class
        """
        super().__init__(config, log_dir, **kwargs)
        self.analyze_gradients = analyze_gradients
        self.analyze_weights = analyze_weights
        self.analyze_activations = analyze_activations

        self.gradient_stats = []
        self.weight_stats = []
        self.activation_hooks = []

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        """Analyze model after each training batch."""
        if batch_idx % 100 == 0:  # Sample periodically
            if self.analyze_gradients:
                self._analyze_gradients(pl_module)
            if self.analyze_weights:
                self._analyze_weights(pl_module)

    def _analyze_gradients(self, pl_module: pl.LightningModule) -> None:
        """Analyze gradient statistics."""
        grad_stats = {}

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_stats[name] = {
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "max": grad.max().item(),
                    "min": grad.min().item(),
                    "norm": grad.norm().item()
                }

        self.gradient_stats.append(grad_stats)

    def _analyze_weights(self, pl_module: pl.LightningModule) -> None:
        """Analyze weight statistics."""
        weight_stats = {}

        for name, param in pl_module.named_parameters():
            if param.data is not None:
                weight = param.data
                weight_stats[name] = {
                    "mean": weight.mean().item(),
                    "std": weight.std().item(),
                    "max": weight.max().item(),
                    "min": weight.min().item(),
                    "norm": weight.norm().item()
                }

        self.weight_stats.append(weight_stats)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save analysis results at the end of training."""
        super().on_train_end(trainer, pl_module)

        if self.gradient_stats:
            grad_file = self.log_dir / "gradient_analysis.json"
            with open(grad_file, "w") as f:
                json.dump(self.gradient_stats, f, indent=2, default=str)

        if self.weight_stats:
            weight_file = self.log_dir / "weight_analysis.json"
            with open(weight_file, "w") as f:
                json.dump(self.weight_stats, f, indent=2, default=str)