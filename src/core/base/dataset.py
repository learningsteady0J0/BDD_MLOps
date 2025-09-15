"""Base data module class for all datasets in the framework."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch


class BaseDataModule(pl.LightningDataModule, ABC):
    """
    Abstract base class for all data modules in the framework.

    This class provides a consistent interface for data handling
    across different domains and data types.

    Attributes:
        data_dir: Root directory for data storage
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs
    ):
        """
        Initialize the base data module.

        Args:
            data_dir: Directory containing the data
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
            **kwargs: Additional arguments
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers and num_workers > 0

        # Datasets
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # Save hyperparameters
        self.save_hyperparameters()

    @abstractmethod
    def prepare_data(self) -> None:
        """
        Download and prepare data. Called only on rank 0 in distributed training.

        This method should handle:
        - Downloading raw data
        - Extracting archives
        - Initial preprocessing that needs to be done once
        """
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for each stage.

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict')
        """
        pass

    @property
    @abstractmethod
    def num_classes(self) -> Optional[int]:
        """Return the number of classes for classification tasks."""
        pass

    @property
    @abstractmethod
    def data_shape(self) -> tuple:
        """Return the shape of a single data sample."""
        pass

    def train_dataloader(self) -> DataLoader:
        """
        Create the training dataloader.

        Returns:
            DataLoader for training data
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation dataloader.

        Returns:
            DataLoader for validation data
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create the test dataloader.

        Returns:
            DataLoader for test data
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Create the prediction dataloader.

        Returns:
            DataLoader for prediction data
        """
        # Default to test dataloader for predictions
        return self.test_dataloader()

    def get_data_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'data_dir': str(self.data_dir),
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'data_shape': self.data_shape
        }

        if self.train_dataset:
            stats['train_size'] = len(self.train_dataset)
        if self.val_dataset:
            stats['val_size'] = len(self.val_dataset)
        if self.test_dataset:
            stats['test_size'] = len(self.test_dataset)
        if self.num_classes:
            stats['num_classes'] = self.num_classes

        return stats

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Clean up after training/testing.

        Args:
            stage: Current stage
        """
        # Clean up any temporary files or resources
        pass


class BaseVisionDataModule(BaseDataModule):
    """Base class for vision data modules."""

    def __init__(
        self,
        image_size: tuple = (224, 224),
        normalize: bool = True,
        augment: bool = True,
        **kwargs
    ):
        """
        Initialize vision data module.

        Args:
            image_size: Target image size (height, width)
            normalize: Whether to normalize images
            augment: Whether to apply data augmentation
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment

    @property
    def data_shape(self) -> tuple:
        """Return image data shape."""
        return (3, *self.image_size)  # Assuming RGB images

    @abstractmethod
    def get_transforms(self, stage: str):
        """Get transforms for the given stage."""
        pass


class BaseNLPDataModule(BaseDataModule):
    """Base class for NLP data modules."""

    def __init__(
        self,
        max_length: int = 512,
        tokenizer: Optional[Any] = None,
        vocab_size: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize NLP data module.

        Args:
            max_length: Maximum sequence length
            tokenizer: Tokenizer to use
            vocab_size: Size of vocabulary
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    @property
    def data_shape(self) -> tuple:
        """Return text data shape."""
        return (self.max_length,)

    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize input text."""
        pass


class BaseTimeSeriesDataModule(BaseDataModule):
    """Base class for time series data modules."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        **kwargs
    ):
        """
        Initialize time series data module.

        Args:
            sequence_length: Length of input sequences
            prediction_length: Length of prediction horizon
            num_features: Number of features per time step
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_features = num_features

    @property
    def data_shape(self) -> tuple:
        """Return time series data shape."""
        return (self.sequence_length, self.num_features)

    @property
    def num_classes(self) -> Optional[int]:
        """Time series typically doesn't have classes."""
        return None

    @abstractmethod
    def create_sequences(self, data: torch.Tensor) -> tuple:
        """Create sequences from time series data."""
        pass