"""Interface definitions and contracts for the framework components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable
import torch
from torch import nn
from dataclasses import dataclass
from enum import Enum


class ModelState(Enum):
    """Enumeration of model states."""
    CREATED = "created"
    INITIALIZED = "initialized"
    TRAINING = "training"
    VALIDATING = "validating"
    TESTING = "testing"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"


class DataSplit(Enum):
    """Enumeration of data splits."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PREDICT = "predict"


@runtime_checkable
class ITrainable(Protocol):
    """Protocol for trainable components."""

    def train(self) -> None:
        """Set component to training mode."""
        ...

    def eval(self) -> None:
        """Set component to evaluation mode."""
        ...

    def parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters."""
        ...

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary."""
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        ...


@runtime_checkable
class ISerializable(Protocol):
    """Protocol for serializable components."""

    def save(self, path: str) -> None:
        """Save component to disk."""
        ...

    def load(self, path: str) -> None:
        """Load component from disk."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ISerializable':
        """Create from dictionary representation."""
        ...


@runtime_checkable
class IConfigurable(Protocol):
    """Protocol for configurable components."""

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure component with given configuration."""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        ...

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        ...


@runtime_checkable
class IMonitorable(Protocol):
    """Protocol for monitorable components."""

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        ...

    def reset_metrics(self) -> None:
        """Reset metrics."""
        ...

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics."""
        ...


@dataclass
class ModelContract:
    """Contract for model implementations."""

    # Required methods
    required_methods: List[str] = None

    # Required attributes
    required_attributes: List[str] = None

    # Input/output specifications
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None

    # Performance requirements
    max_inference_time_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None

    # Compatibility
    min_pytorch_version: Optional[str] = None
    cuda_required: bool = False

    def __post_init__(self):
        """Initialize default values."""
        if self.required_methods is None:
            self.required_methods = ['forward', 'training_step', 'validation_step']
        if self.required_attributes is None:
            self.required_attributes = ['model_config', 'loss_fn']

    def validate(self, model_class: type) -> bool:
        """
        Validate that a model class meets the contract.

        Args:
            model_class: Model class to validate

        Returns:
            True if contract is satisfied

        Raises:
            ValueError: If contract is violated
        """
        # Check required methods
        for method in self.required_methods:
            if not hasattr(model_class, method):
                raise ValueError(f"Model missing required method: {method}")

        # Check if methods are callable
        for method in self.required_methods:
            if not callable(getattr(model_class, method, None)):
                raise ValueError(f"Required method {method} is not callable")

        return True


@dataclass
class DataContract:
    """Contract for data module implementations."""

    # Required methods
    required_methods: List[str] = None

    # Data specifications
    min_samples: Optional[int] = None
    max_samples: Optional[int] = None

    # Feature specifications
    num_features: Optional[int] = None
    feature_types: Optional[Dict[str, type]] = None

    # Split requirements
    train_ratio: Optional[float] = None
    val_ratio: Optional[float] = None
    test_ratio: Optional[float] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.required_methods is None:
            self.required_methods = [
                'prepare_data',
                'setup',
                'train_dataloader',
                'val_dataloader'
            ]

    def validate(self, data_class: type) -> bool:
        """
        Validate that a data class meets the contract.

        Args:
            data_class: Data class to validate

        Returns:
            True if contract is satisfied

        Raises:
            ValueError: If contract is violated
        """
        # Check required methods
        for method in self.required_methods:
            if not hasattr(data_class, method):
                raise ValueError(f"DataModule missing required method: {method}")

        return True


class IPlugin(ABC):
    """Abstract interface for all plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Get plugin version."""
        pass

    @property
    @abstractmethod
    def domain(self) -> str:
        """Get plugin domain (vision, nlp, timeseries, etc.)."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate plugin configuration and dependencies."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass


class IModelPlugin(IPlugin):
    """Interface for model plugins."""

    @abstractmethod
    def get_model_class(self) -> type:
        """Get the model class."""
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration."""
        pass

    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported tasks."""
        pass


class IDataPlugin(IPlugin):
    """Interface for data plugins."""

    @abstractmethod
    def get_data_class(self) -> type:
        """Get the data module class."""
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default data configuration."""
        pass

    @abstractmethod
    def get_data_stats(self) -> Dict[str, Any]:
        """Get data statistics."""
        pass


class ITransformPlugin(IPlugin):
    """Interface for transform plugins."""

    @abstractmethod
    def get_transform_class(self) -> type:
        """Get the transform class."""
        pass

    @abstractmethod
    def get_supported_inputs(self) -> List[str]:
        """Get list of supported input types."""
        pass

    @abstractmethod
    def apply(self, data: Any) -> Any:
        """Apply the transform to data."""
        pass


class ICallbackPlugin(IPlugin):
    """Interface for callback plugins."""

    @abstractmethod
    def get_callback_class(self) -> type:
        """Get the callback class."""
        pass

    @abstractmethod
    def get_hook_points(self) -> List[str]:
        """Get list of hook points this callback uses."""
        pass

    @abstractmethod
    def get_priority(self) -> int:
        """Get callback priority (lower executes first)."""
        pass


class IMetricPlugin(IPlugin):
    """Interface for metric plugins."""

    @abstractmethod
    def get_metric_class(self) -> type:
        """Get the metric class."""
        pass

    @abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute the metric value."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset metric state."""
        pass


class IOptimizerPlugin(IPlugin):
    """Interface for optimizer plugins."""

    @abstractmethod
    def get_optimizer_class(self) -> type:
        """Get the optimizer class."""
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default optimizer parameters."""
        pass

    @abstractmethod
    def create_optimizer(self, parameters: List[nn.Parameter], **kwargs) -> torch.optim.Optimizer:
        """Create optimizer instance."""
        pass


class ISchedulerPlugin(IPlugin):
    """Interface for scheduler plugins."""

    @abstractmethod
    def get_scheduler_class(self) -> type:
        """Get the scheduler class."""
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default scheduler parameters."""
        pass

    @abstractmethod
    def create_scheduler(self, optimizer: torch.optim.Optimizer, **kwargs) -> Any:
        """Create scheduler instance."""
        pass


class IExperimentTracker(ABC):
    """Interface for experiment tracking."""

    @abstractmethod
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new experiment run."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        pass

    @abstractmethod
    def log_artifact(self, artifact_path: str, artifact_type: Optional[str] = None) -> None:
        """Log an artifact."""
        pass

    @abstractmethod
    def end_run(self) -> None:
        """End the current run."""
        pass

    @abstractmethod
    def get_run_id(self) -> str:
        """Get current run ID."""
        pass


class IModelRegistry(ABC):
    """Interface for model registry."""

    @abstractmethod
    def register_model(self, model: nn.Module, name: str, version: str, metadata: Dict[str, Any]) -> str:
        """Register a model."""
        pass

    @abstractmethod
    def get_model(self, name: str, version: Optional[str] = None) -> nn.Module:
        """Get a registered model."""
        pass

    @abstractmethod
    def list_models(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered models."""
        pass

    @abstractmethod
    def delete_model(self, name: str, version: str) -> bool:
        """Delete a registered model."""
        pass

    @abstractmethod
    def promote_model(self, name: str, version: str, stage: str) -> bool:
        """Promote model to a stage (staging, production)."""
        pass