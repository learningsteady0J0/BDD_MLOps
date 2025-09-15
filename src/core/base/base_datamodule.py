"""Base Vision DataModule with augmentation and preprocessing pipeline."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from omegaconf import DictConfig
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class BaseVisionDataModule(pl.LightningDataModule, ABC):
    """
    Abstract base class for vision data modules.

    This class provides:
    - Standardized data loading pipeline
    - Configurable augmentation strategies
    - Multi-stage preprocessing
    - Automatic dataset splitting
    - Memory-efficient data loading
    """

    def __init__(
        self,
        config: DictConfig,
        data_dir: Union[str, Path] = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        image_size: Union[int, tuple] = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        augmentation_config: Optional[Dict[str, Any]] = None,
        val_split: float = 0.2,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize the base vision data module.

        Args:
            config: Hydra configuration object
            data_dir: Root directory for datasets
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU transfer
            persistent_workers: Whether to keep workers alive between epochs
            image_size: Target image size (int or tuple)
            mean: Normalization mean values
            std: Normalization std values
            train_transforms: Custom training transforms
            val_transforms: Custom validation transforms
            test_transforms: Custom test transforms
            augmentation_config: Augmentation configuration
            val_split: Validation split ratio
            test_split: Test split ratio
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.save_hyperparameters(ignore=["config"])

        self.config = config
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        # Image preprocessing
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.mean = mean or [0.485, 0.456, 0.406]  # ImageNet defaults
        self.std = std or [0.229, 0.224, 0.225]

        # Data splits
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        # Transforms
        self.train_transforms = train_transforms or self._default_train_transforms(augmentation_config)
        self.val_transforms = val_transforms or self._default_val_transforms()
        self.test_transforms = test_transforms or self._default_test_transforms()

        # Datasets (to be set in setup)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # Class information
        self.num_classes: Optional[int] = None
        self.classes: Optional[List[str]] = None
        self.class_weights: Optional[torch.Tensor] = None

    def _default_train_transforms(self, augmentation_config: Optional[Dict[str, Any]] = None) -> Callable:
        """Create default training transforms with augmentation."""
        augmentation_config = augmentation_config or {}

        transforms = []

        # Resize and basic preprocessing
        transforms.append(T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR))

        # Data augmentation based on configuration
        if augmentation_config.get("random_crop", False):
            transforms.append(T.RandomCrop(self.image_size, padding=4))

        if augmentation_config.get("horizontal_flip", True):
            transforms.append(T.RandomHorizontalFlip(p=0.5))

        if augmentation_config.get("vertical_flip", False):
            transforms.append(T.RandomVerticalFlip(p=0.5))

        if augmentation_config.get("rotation", False):
            degrees = augmentation_config.get("rotation_degrees", 15)
            transforms.append(T.RandomRotation(degrees))

        if augmentation_config.get("color_jitter", False):
            transforms.append(
                T.ColorJitter(
                    brightness=augmentation_config.get("brightness", 0.2),
                    contrast=augmentation_config.get("contrast", 0.2),
                    saturation=augmentation_config.get("saturation", 0.2),
                    hue=augmentation_config.get("hue", 0.1),
                )
            )

        if augmentation_config.get("random_erasing", False):
            transforms.append(
                T.RandomErasing(
                    p=augmentation_config.get("erasing_p", 0.5),
                    scale=augmentation_config.get("erasing_scale", (0.02, 0.33)),
                    ratio=augmentation_config.get("erasing_ratio", (0.3, 3.3)),
                )
            )

        if augmentation_config.get("auto_augment", False):
            policy = augmentation_config.get("auto_augment_policy", "imagenet")
            if policy == "imagenet":
                transforms.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET))
            elif policy == "cifar10":
                transforms.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))

        # Convert to tensor and normalize
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(mean=self.mean, std=self.std))

        if augmentation_config.get("mixup", False) or augmentation_config.get("cutmix", False):
            # These will be handled in the collate_fn or training loop
            pass

        return T.Compose(transforms)

    def _default_val_transforms(self) -> Callable:
        """Create default validation transforms."""
        return T.Compose([
            T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])

    def _default_test_transforms(self) -> Callable:
        """Create default test transforms."""
        return self._default_val_transforms()

    @abstractmethod
    def prepare_data(self) -> None:
        """Download and prepare data. Called only on 1 GPU/process."""
        raise NotImplementedError("Subclasses must implement prepare_data")

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for each stage."""
        raise NotImplementedError("Subclasses must implement setup")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return self._create_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return self._create_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return self._create_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def _create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        """Create a dataloader with common settings."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=drop_last,
            collate_fn=self.collate_fn if hasattr(self, "collate_fn") else None,
        )

    def split_dataset(self, dataset: Dataset) -> tuple:
        """Split dataset into train/val/test sets."""
        total_size = len(dataset)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size - test_size

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

        return train_dataset, val_dataset, test_dataset

    def get_num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        if self.num_classes is None:
            raise ValueError("num_classes not set. Call setup() first.")
        return self.num_classes

    def get_class_weights(self, samples_per_class: Optional[List[int]] = None) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.

        Args:
            samples_per_class: Number of samples per class

        Returns:
            Tensor of class weights
        """
        if samples_per_class is None:
            # Calculate from training dataset
            if self.train_dataset is None:
                raise ValueError("Train dataset not initialized. Call setup() first.")

            samples_per_class = self._count_samples_per_class()

        total_samples = sum(samples_per_class)
        weights = [total_samples / (len(samples_per_class) * count) for count in samples_per_class]
        return torch.tensor(weights, dtype=torch.float32)

    def _count_samples_per_class(self) -> List[int]:
        """Count samples per class in training dataset."""
        counts = [0] * self.num_classes
        for _, label in self.train_dataset:
            counts[label] += 1
        return counts

    def visualize_batch(self, batch: tuple, num_samples: int = 8) -> None:
        """Visualize a batch of samples."""
        import matplotlib.pyplot as plt
        import numpy as np

        images, labels = batch
        images = images[:num_samples].cpu()

        # Denormalize
        for i in range(3):
            images[:, i, :, :] = images[:, i, :, :] * self.std[i] + self.mean[i]

        images = torch.clamp(images, 0, 1)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()

        for i, (img, label) in enumerate(zip(images, labels[:num_samples])):
            img = img.permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {label.item() if hasattr(label, 'item') else label}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "num_train_samples": len(self.train_dataset) if self.train_dataset else 0,
            "num_val_samples": len(self.val_dataset) if self.val_dataset else 0,
            "num_test_samples": len(self.test_dataset) if self.test_dataset else 0,
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "normalization": {"mean": self.mean, "std": self.std},
        }

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Apply GPU/TPU transforms before batch transfer."""
        # Can be overridden for GPU-accelerated augmentations
        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Apply GPU/TPU transforms after batch transfer."""
        # Can be overridden for GPU-accelerated augmentations
        return batch