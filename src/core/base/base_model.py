"""Base Vision Model class with PyTorch Lightning integration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, List
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from omegaconf import DictConfig
import torchmetrics


class BaseVisionModel(pl.LightningModule, ABC):
    """
    Abstract base class for all vision models in the framework.

    This class provides a standard interface for vision models with support for:
    - Multiple vision tasks (classification, detection, segmentation)
    - Transfer learning and fine-tuning
    - Flexible backbone/head architecture
    - Automatic metric tracking
    - MLflow integration hooks
    """

    def __init__(
        self,
        config: DictConfig,
        num_classes: Optional[int] = None,
        task_type: str = "classification",
        backbone_name: Optional[str] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the base vision model.

        Args:
            config: Hydra configuration object
            num_classes: Number of output classes
            task_type: Type of vision task ('classification', 'detection', 'segmentation')
            backbone_name: Name of the backbone architecture
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights during training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            scheduler_config: Configuration for learning rate scheduler
        """
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.num_classes = num_classes
        self.task_type = task_type
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}

        # Initialize components
        self.backbone = self._build_backbone()
        self.head = self._build_head()

        if freeze_backbone:
            self._freeze_backbone()

        # Initialize metrics
        self.metrics = self._setup_metrics()

        # Loss function
        self.criterion = self._setup_loss()

        # Hooks for extensibility
        self._register_hooks()

    @abstractmethod
    def _build_backbone(self) -> nn.Module:
        """Build the backbone network."""
        raise NotImplementedError("Subclasses must implement _build_backbone")

    @abstractmethod
    def _build_head(self) -> nn.Module:
        """Build the task-specific head."""
        raise NotImplementedError("Subclasses must implement _build_head")

    @abstractmethod
    def _setup_loss(self) -> nn.Module:
        """Setup the loss function for the task."""
        raise NotImplementedError("Subclasses must implement _setup_loss")

    def _setup_metrics(self) -> nn.ModuleDict:
        """Setup metrics for tracking."""
        metrics = nn.ModuleDict()

        if self.task_type == "classification":
            metrics["train_acc"] = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=self.num_classes
            )
            metrics["val_acc"] = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=self.num_classes
            )
            metrics["train_f1"] = torchmetrics.F1Score(
                task="multiclass",
                num_classes=self.num_classes
            )
            metrics["val_f1"] = torchmetrics.F1Score(
                task="multiclass",
                num_classes=self.num_classes
            )
        elif self.task_type == "detection":
            # Detection metrics would go here
            pass
        elif self.task_type == "segmentation":
            metrics["train_iou"] = torchmetrics.JaccardIndex(
                task="multiclass",
                num_classes=self.num_classes
            )
            metrics["val_iou"] = torchmetrics.JaccardIndex(
                task="multiclass",
                num_classes=self.num_classes
            )

        return metrics

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _register_hooks(self) -> None:
        """Register hooks for extensibility."""
        self.hooks = {
            "before_forward": [],
            "after_forward": [],
            "before_backward": [],
            "after_backward": [],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Execute before_forward hooks
        for hook in self.hooks.get("before_forward", []):
            x = hook(x)

        # Extract features from backbone
        features = self.backbone(x)

        # Pass through task-specific head
        output = self.head(features)

        # Execute after_forward hooks
        for hook in self.hooks.get("after_forward", []):
            output = hook(output)

        return output

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        if self.task_type == "classification" and "train_acc" in self.metrics:
            self.metrics["train_acc"](y_hat, y)
            self.log("train/acc", self.metrics["train_acc"], prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Log metrics
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if self.task_type == "classification" and "val_acc" in self.metrics:
            self.metrics["val_acc"](y_hat, y)
            self.log("val/acc", self.metrics["val_acc"], prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> Union[Optimizer, Dict[str, Any]]:
        """Configure optimizers and schedulers."""
        # Separate parameters for different learning rates
        params = self._get_parameter_groups()

        # Create optimizer
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Create scheduler if configured
        if self.scheduler_config:
            scheduler = self._create_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    **self.scheduler_config
                }
            }

        return optimizer

    def _get_parameter_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups for optimizer."""
        # Default: all parameters with same learning rate
        return [{"params": self.parameters()}]

    def _create_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        """Create learning rate scheduler."""
        scheduler_type = self.scheduler_config.get("type", "cosine")

        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get("T_max", 100),
                eta_min=self.scheduler_config.get("eta_min", 1e-6)
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get("step_size", 30),
                gamma=self.scheduler_config.get("gamma", 0.1)
            )
        elif scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_config.get("factor", 0.1),
                patience=self.scheduler_config.get("patience", 10)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def on_train_epoch_end(self) -> None:
        """Hook called at the end of training epoch."""
        # Reset metrics for next epoch
        for key, metric in self.metrics.items():
            if "train" in key:
                metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Hook called at the end of validation epoch."""
        # Reset metrics for next epoch
        for key, metric in self.metrics.items():
            if "val" in key:
                metric.reset()

    def register_hook(self, hook_name: str, hook_fn: callable) -> None:
        """Register a custom hook."""
        if hook_name in self.hooks:
            self.hooks[hook_name].append(hook_fn)
        else:
            raise ValueError(f"Unknown hook name: {hook_name}")

    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone only."""
        return self.backbone(x)

    def load_backbone_weights(self, checkpoint_path: Union[str, Path]) -> None:
        """Load pretrained backbone weights."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint, strict=False)

    def summary(self) -> str:
        """Get model summary."""
        from torchinfo import summary
        return summary(self, input_size=(1, 3, 224, 224), device=self.device)