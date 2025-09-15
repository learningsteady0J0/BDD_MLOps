"""Trainer interfaces and contracts for the Vision framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig


class ITrainer(ABC):
    """Interface for training orchestration."""

    @abstractmethod
    def fit(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule
    ) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def validate(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule
    ) -> Dict[str, float]:
        """Validate the model."""
        pass

    @abstractmethod
    def test(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule
    ) -> Dict[str, float]:
        """Test the model."""
        pass

    @abstractmethod
    def predict(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule
    ) -> List[torch.Tensor]:
        """Run inference."""
        pass

    @property
    @abstractmethod
    def callbacks(self) -> List[Callback]:
        """Get list of callbacks."""
        pass

    @abstractmethod
    def save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Save training checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Load training checkpoint."""
        pass


class IOptimizer(ABC):
    """Interface for optimizers."""

    @abstractmethod
    def step(self) -> None:
        """Perform optimization step."""
        pass

    @abstractmethod
    def zero_grad(self) -> None:
        """Zero gradients."""
        pass

    @property
    @abstractmethod
    def param_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups."""
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state."""
        pass


class IScheduler(ABC):
    """Interface for learning rate schedulers."""

    @abstractmethod
    def step(self, metrics: Optional[float] = None) -> None:
        """Update learning rate."""
        pass

    @abstractmethod
    def get_last_lr(self) -> List[float]:
        """Get last learning rate."""
        pass

    @property
    @abstractmethod
    def optimizer(self) -> IOptimizer:
        """Get associated optimizer."""
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        pass


class TrainerContract:
    """Contract validation for training components."""

    @staticmethod
    def validate_trainer(trainer: pl.Trainer) -> bool:
        """Validate trainer contract."""
        assert hasattr(trainer, "fit"), "Trainer must have fit method"
        assert hasattr(trainer, "validate"), "Trainer must have validate method"
        assert hasattr(trainer, "test"), "Trainer must have test method"
        assert hasattr(trainer, "predict"), "Trainer must have predict method"
        assert hasattr(trainer, "callbacks"), "Trainer must have callbacks property"
        return True

    @staticmethod
    def validate_optimizer(optimizer: torch.optim.Optimizer) -> bool:
        """Validate optimizer contract."""
        assert hasattr(optimizer, "step"), "Optimizer must have step method"
        assert hasattr(optimizer, "zero_grad"), "Optimizer must have zero_grad method"
        assert hasattr(optimizer, "param_groups"), "Optimizer must have param_groups"
        assert hasattr(optimizer, "state_dict"), "Optimizer must have state_dict method"
        return True

    @staticmethod
    def validate_scheduler(scheduler: Any) -> bool:
        """Validate scheduler contract."""
        assert hasattr(scheduler, "step"), "Scheduler must have step method"
        assert hasattr(scheduler, "get_last_lr"), "Scheduler must have get_last_lr method"
        assert hasattr(scheduler, "state_dict"), "Scheduler must have state_dict method"
        return True

    @staticmethod
    def validate_training_config(config: DictConfig) -> bool:
        """Validate training configuration."""
        required_keys = ["trainer", "model", "data"]
        for key in required_keys:
            assert key in config, f"Config must have '{key}' section"

        # Validate trainer config
        trainer_config = config.trainer
        assert "max_epochs" in trainer_config, "Trainer config must have max_epochs"
        assert trainer_config.max_epochs > 0, "max_epochs must be positive"

        # Validate model config
        model_config = config.model
        assert "learning_rate" in model_config, "Model config must have learning_rate"
        assert model_config.learning_rate > 0, "learning_rate must be positive"

        return True