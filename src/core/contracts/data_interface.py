"""Data interfaces and contracts for the Vision framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, List, Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class IDataModule(ABC):
    """Interface for data modules."""

    @abstractmethod
    def prepare_data(self) -> None:
        """Download and prepare data."""
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for train/val/test."""
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Get number of classes."""
        pass

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, int]:
        """Get expected input size."""
        pass


class IDataset(ABC):
    """Interface for vision datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Get dataset size."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get a sample from the dataset."""
        pass

    @property
    @abstractmethod
    def classes(self) -> List[str]:
        """Get list of class names."""
        pass

    @abstractmethod
    def get_sample_weight(self, idx: int) -> float:
        """Get weight for a specific sample."""
        pass


class ITransform(ABC):
    """Interface for data transforms."""

    @abstractmethod
    def __call__(self, sample: Any) -> Any:
        """Apply transform to sample."""
        pass

    @property
    @abstractmethod
    def deterministic(self) -> bool:
        """Whether transform is deterministic."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get transform parameters."""
        pass


class IImageDataset(IDataset):
    """Interface for image datasets."""

    @abstractmethod
    def load_image(self, path: Union[str, Path]) -> Union[Image.Image, np.ndarray]:
        """Load an image from path."""
        pass

    @abstractmethod
    def load_annotation(self, path: Union[str, Path]) -> Any:
        """Load annotation for an image."""
        pass

    @property
    @abstractmethod
    def image_paths(self) -> List[Path]:
        """Get list of all image paths."""
        pass


class IVideoDataset(IDataset):
    """Interface for video datasets."""

    @abstractmethod
    def load_video(self, path: Union[str, Path]) -> torch.Tensor:
        """Load a video from path."""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Get video frame rate."""
        pass

    @property
    @abstractmethod
    def num_frames(self) -> int:
        """Get number of frames to sample."""
        pass


class DataContract:
    """Contract validation for data components."""

    @staticmethod
    def validate_datamodule(datamodule: Any) -> bool:
        """Validate datamodule contract."""
        assert hasattr(datamodule, "prepare_data"), "Must have prepare_data method"
        assert hasattr(datamodule, "setup"), "Must have setup method"
        assert hasattr(datamodule, "train_dataloader"), "Must have train_dataloader method"
        assert hasattr(datamodule, "val_dataloader"), "Must have val_dataloader method"
        assert hasattr(datamodule, "num_classes"), "Must have num_classes property"
        return True

    @staticmethod
    def validate_dataset(dataset: Dataset) -> bool:
        """Validate dataset contract."""
        assert hasattr(dataset, "__len__"), "Must have __len__ method"
        assert hasattr(dataset, "__getitem__"), "Must have __getitem__ method"
        assert len(dataset) > 0, "Dataset must not be empty"

        # Test getting a sample
        sample = dataset[0]
        assert isinstance(sample, (tuple, list)), "Sample must be tuple or list"
        assert len(sample) >= 2, "Sample must have at least input and target"

        return True

    @staticmethod
    def validate_transform(transform: Callable) -> bool:
        """Validate transform contract."""
        assert callable(transform), "Transform must be callable"
        return True

    @staticmethod
    def validate_batch(batch: Tuple[torch.Tensor, torch.Tensor]) -> bool:
        """Validate a data batch."""
        assert isinstance(batch, (tuple, list)), "Batch must be tuple or list"
        assert len(batch) >= 2, "Batch must have inputs and targets"

        inputs, targets = batch[0], batch[1]
        assert isinstance(inputs, torch.Tensor), "Inputs must be tensor"
        assert isinstance(targets, (torch.Tensor, list)), "Targets must be tensor or list"

        # Check dimensions
        assert inputs.dim() >= 3, "Input must have at least 3 dimensions"

        return True