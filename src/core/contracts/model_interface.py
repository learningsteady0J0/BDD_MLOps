"""Model interfaces and contracts for the Vision framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, List
from enum import Enum

import torch
import torch.nn as nn
from omegaconf import DictConfig


class TaskType(Enum):
    """Enumeration of supported vision tasks."""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    KEYPOINT_DETECTION = "keypoint_detection"
    DEPTH_ESTIMATION = "depth_estimation"
    IMAGE_GENERATION = "image_generation"


class IVisionModel(ABC):
    """Interface for all vision models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def get_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate loss for the predictions."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        pass

    @abstractmethod
    def configure_optimizers(self) -> Any:
        """Configure optimizers and schedulers."""
        pass

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Get the task type this model is designed for."""
        pass


class IBackbone(ABC):
    """Interface for model backbones (feature extractors)."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Extract features from input.

        Args:
            x: Input tensor

        Returns:
            Features or list of features (for FPN-style backbones)
        """
        pass

    @property
    @abstractmethod
    def output_channels(self) -> Union[int, List[int]]:
        """Get number of output channels."""
        pass

    @property
    @abstractmethod
    def output_stride(self) -> int:
        """Get the output stride of the backbone."""
        pass

    @abstractmethod
    def freeze(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone parameters."""
        pass

    @abstractmethod
    def get_stages(self) -> List[nn.Module]:
        """Get list of backbone stages for feature pyramid."""
        pass


class IHead(ABC):
    """Interface for task-specific heads."""

    @abstractmethod
    def forward(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Process features to produce task-specific output.

        Args:
            features: Features from backbone

        Returns:
            Task-specific predictions
        """
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Get number of output classes."""
        pass

    @abstractmethod
    def get_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate task-specific loss."""
        pass


class IClassificationHead(IHead):
    """Interface for classification heads."""

    @abstractmethod
    def get_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Get raw logits before activation."""
        pass

    @abstractmethod
    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        pass


class IDetectionHead(IHead):
    """Interface for object detection heads."""

    @abstractmethod
    def get_boxes(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Get bounding box predictions."""
        pass

    @abstractmethod
    def get_scores(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Get objectness/class scores."""
        pass

    @abstractmethod
    def post_process(
        self,
        predictions: Dict[str, torch.Tensor],
        image_sizes: List[Tuple[int, int]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process predictions (NMS, etc.)."""
        pass


class ISegmentationHead(IHead):
    """Interface for segmentation heads."""

    @abstractmethod
    def get_masks(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Get segmentation masks."""
        pass

    @abstractmethod
    def upsample(self, masks: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Upsample masks to target size."""
        pass


class ModelContract:
    """
    Contract validation for vision models.

    Ensures models conform to expected interfaces and behaviors.
    """

    @staticmethod
    def validate_model(model: nn.Module, task_type: TaskType) -> bool:
        """
        Validate that a model conforms to its task-specific contract.

        Args:
            model: Model to validate
            task_type: Expected task type

        Returns:
            True if valid

        Raises:
            AssertionError: If validation fails
        """
        # Check basic interface
        assert hasattr(model, "forward"), "Model must have forward method"
        assert hasattr(model, "training_step"), "Model must have training_step method"
        assert hasattr(model, "validation_step"), "Model must have validation_step method"
        assert hasattr(model, "configure_optimizers"), "Model must have configure_optimizers method"

        # Task-specific validation
        if task_type == TaskType.CLASSIFICATION:
            assert hasattr(model, "num_classes"), "Classification model must have num_classes"

        elif task_type == TaskType.DETECTION:
            assert hasattr(model, "get_boxes"), "Detection model must have get_boxes method"
            assert hasattr(model, "post_process"), "Detection model must have post_process method"

        elif task_type == TaskType.SEGMENTATION:
            assert hasattr(model, "get_masks"), "Segmentation model must have get_masks method"

        return True

    @staticmethod
    def validate_backbone(backbone: nn.Module) -> bool:
        """Validate backbone contract."""
        assert hasattr(backbone, "forward"), "Backbone must have forward method"
        assert hasattr(backbone, "output_channels"), "Backbone must have output_channels property"
        assert hasattr(backbone, "output_stride"), "Backbone must have output_stride property"
        return True

    @staticmethod
    def validate_head(head: nn.Module, task_type: TaskType) -> bool:
        """Validate head contract."""
        assert hasattr(head, "forward"), "Head must have forward method"
        assert hasattr(head, "num_classes"), "Head must have num_classes property"
        assert hasattr(head, "get_loss"), "Head must have get_loss method"

        if task_type == TaskType.CLASSIFICATION:
            assert isinstance(head, IClassificationHead), "Must implement IClassificationHead"
        elif task_type == TaskType.DETECTION:
            assert isinstance(head, IDetectionHead), "Must implement IDetectionHead"
        elif task_type == TaskType.SEGMENTATION:
            assert isinstance(head, ISegmentationHead), "Must implement ISegmentationHead"

        return True