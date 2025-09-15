"""Base model class for all AI models in the framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Metric


class BaseModel(pl.LightningModule, ABC):
    """
    Abstract base class for all models in the framework.

    This class provides a consistent interface for model implementation
    across different domains (vision, NLP, time series, etc.).

    Attributes:
        model_config: Configuration dictionary for the model
        loss_fn: Loss function for training
        metrics: Dictionary of metrics to track
        learning_rate: Base learning rate for optimization
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 1e-3,
        **kwargs
    ):
        """
        Initialize the base model.

        Args:
            model_config: Model-specific configuration
            loss_fn: Loss function to use
            learning_rate: Learning rate for optimizer
            **kwargs: Additional arguments
        """
        super().__init__()
        self.model_config = model_config
        self.loss_fn = loss_fn or self._default_loss()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        # Initialize metrics
        self.metrics = self._setup_metrics()

        # Build model architecture
        self.model = self._build_model()

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Build the model architecture."""
        pass

    @abstractmethod
    def _default_loss(self) -> nn.Module:
        """Return the default loss function for this model type."""
        pass

    @abstractmethod
    def _setup_metrics(self) -> Dict[str, Metric]:
        """Setup metrics for training and validation."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step logic.

        Args:
            batch: Input batch containing features and targets
            batch_idx: Index of the current batch

        Returns:
            Loss value for backpropagation
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self._update_metrics('train', y_hat, y)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> None:
        """
        Validation step logic.

        Args:
            batch: Input batch containing features and targets
            batch_idx: Index of the current batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self._update_metrics('val', y_hat, y)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> None:
        """
        Test step logic.

        Args:
            batch: Input batch containing features and targets
            batch_idx: Index of the current batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Log metrics
        self.log('test_loss', loss)
        self._update_metrics('test', y_hat, y)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def _update_metrics(
        self,
        stage: str,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> None:
        """
        Update metrics for the given stage.

        Args:
            stage: Training stage ('train', 'val', 'test')
            preds: Model predictions
            targets: Ground truth targets
        """
        for metric_name, metric in self.metrics.items():
            if stage in metric_name:
                metric.update(preds, targets)
                self.log(
                    metric_name,
                    metric,
                    prog_bar=('val' in metric_name),
                    on_step=False,
                    on_epoch=True
                )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Called when saving a checkpoint.

        Args:
            checkpoint: The checkpoint dictionary
        """
        checkpoint['model_config'] = self.model_config
        checkpoint['model_class'] = self.__class__.__name__

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        **kwargs
    ) -> 'BaseModel':
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            **kwargs: Additional arguments to override

        Returns:
            Loaded model instance
        """
        return super().load_from_checkpoint(checkpoint_path, **kwargs)

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.

        Returns:
            Dictionary containing model information
        """
        return {
            'class': self.__class__.__name__,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            'model_config': self.model_config,
            'learning_rate': self.learning_rate
        }


class BaseVisionModel(BaseModel):
    """Base class for vision models."""

    def __init__(self, num_classes: int, **kwargs):
        """
        Initialize vision model.

        Args:
            num_classes: Number of output classes
            **kwargs: Additional arguments
        """
        self.num_classes = num_classes
        super().__init__(**kwargs)

    def _default_loss(self) -> nn.Module:
        """Default loss for vision models."""
        return nn.CrossEntropyLoss()


class BaseNLPModel(BaseModel):
    """Base class for NLP models."""

    def __init__(self, vocab_size: int, **kwargs):
        """
        Initialize NLP model.

        Args:
            vocab_size: Size of the vocabulary
            **kwargs: Additional arguments
        """
        self.vocab_size = vocab_size
        super().__init__(**kwargs)

    def _default_loss(self) -> nn.Module:
        """Default loss for NLP models."""
        return nn.CrossEntropyLoss(ignore_index=-100)


class BaseTimeSeriesModel(BaseModel):
    """Base class for time series models."""

    def __init__(self, input_size: int, output_size: int, **kwargs):
        """
        Initialize time series model.

        Args:
            input_size: Number of input features
            output_size: Number of output features
            **kwargs: Additional arguments
        """
        self.input_size = input_size
        self.output_size = output_size
        super().__init__(**kwargs)

    def _default_loss(self) -> nn.Module:
        """Default loss for time series models."""
        return nn.MSELoss()