"""CIFAR dataset modules."""

from typing import Optional, Callable
from pathlib import Path

import torch
from torchvision import datasets, transforms
from omegaconf import DictConfig

from src.core.base import BaseVisionDataModule
from src.core.registry import register_datamodule


@register_datamodule(
    name="cifar10",
    dataset_type="image_classification",
    input_size=(32, 32),
    num_classes=10,
    aliases=["CIFAR10", "cifar-10"]
)
class CIFAR10DataModule(BaseVisionDataModule):
    """
    CIFAR-10 dataset module.

    10 classes of 32x32 color images.
    60,000 images total (50,000 train, 10,000 test).
    """

    def __init__(
        self,
        config: DictConfig,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        augment: bool = True,
        **kwargs
    ):
        """Initialize CIFAR-10 data module."""
        super().__init__(
            config=config,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=32,
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 statistics
            std=[0.2470, 0.2435, 0.2616],
            **kwargs
        )

        self.augment = augment
        self.num_classes = 10
        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        # Override transforms for CIFAR-10 specific augmentation
        if augment:
            self.train_transforms = self._cifar_train_transforms()

    def _cifar_train_transforms(self) -> Callable:
        """CIFAR-10 specific training transforms."""
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33))
        ])

    def prepare_data(self) -> None:
        """Download CIFAR-10 dataset."""
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up CIFAR-10 datasets."""
        if stage == "fit" or stage is None:
            # Training dataset
            cifar_train = datasets.CIFAR10(
                self.data_dir,
                train=True,
                transform=self.train_transforms
            )

            # Split into train and validation
            train_size = int(0.9 * len(cifar_train))
            val_size = len(cifar_train) - train_size

            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                cifar_train,
                [train_size, val_size],
                generator=generator
            )

            # Apply different transforms to validation
            self.val_dataset.dataset.transform = self.val_transforms

        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(
                self.data_dir,
                train=False,
                transform=self.test_transforms
            )

    def get_sample_batch(self, n_samples: int = 8) -> tuple:
        """Get a sample batch for visualization."""
        loader = self.train_dataloader()
        batch = next(iter(loader))
        return batch[0][:n_samples], batch[1][:n_samples]


@register_datamodule(
    name="cifar100",
    dataset_type="image_classification",
    input_size=(32, 32),
    num_classes=100,
    aliases=["CIFAR100", "cifar-100"]
)
class CIFAR100DataModule(BaseVisionDataModule):
    """
    CIFAR-100 dataset module.

    100 classes of 32x32 color images.
    60,000 images total (50,000 train, 10,000 test).
    """

    def __init__(
        self,
        config: DictConfig,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        augment: bool = True,
        use_coarse_labels: bool = False,
        **kwargs
    ):
        """Initialize CIFAR-100 data module."""
        super().__init__(
            config=config,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=32,
            mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 statistics
            std=[0.2675, 0.2565, 0.2761],
            **kwargs
        )

        self.augment = augment
        self.use_coarse_labels = use_coarse_labels
        self.num_classes = 20 if use_coarse_labels else 100

        # CIFAR-100 has both fine (100) and coarse (20) labels
        if use_coarse_labels:
            self.classes = self._get_coarse_labels()
        else:
            self.classes = self._get_fine_labels()

        if augment:
            self.train_transforms = self._cifar100_train_transforms()

    def _cifar100_train_transforms(self) -> Callable:
        """CIFAR-100 specific training transforms."""
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            # AutoAugment for CIFAR
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33))
        ])

    def prepare_data(self) -> None:
        """Download CIFAR-100 dataset."""
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up CIFAR-100 datasets."""
        if stage == "fit" or stage is None:
            # Training dataset
            cifar_train = datasets.CIFAR100(
                self.data_dir,
                train=True,
                transform=self.train_transforms
            )

            # Split into train and validation
            train_size = int(0.9 * len(cifar_train))
            val_size = len(cifar_train) - train_size

            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                cifar_train,
                [train_size, val_size],
                generator=generator
            )

            # Apply different transforms to validation
            self.val_dataset.dataset.transform = self.val_transforms

        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR100(
                self.data_dir,
                train=False,
                transform=self.test_transforms
            )

    def _get_fine_labels(self) -> list:
        """Get fine-grained class labels (100 classes)."""
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
            'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
            'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
            'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
            'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
            'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
            'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
            'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
            'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
            'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
            'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
            'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]

    def _get_coarse_labels(self) -> list:
        """Get coarse-grained class labels (20 superclasses)."""
        return [
            'aquatic_mammals', 'fish', 'flowers', 'food_containers',
            'fruit_and_vegetables', 'household_electrical_devices',
            'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
            'large_omnivores_and_herbivores', 'medium_mammals',
            'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
            'trees', 'vehicles_1', 'vehicles_2'
        ]