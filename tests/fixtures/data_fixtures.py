"""Data-related test fixtures for BDD testing."""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch
import tempfile
from typing import Tuple, List, Dict, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, size: int = 100, image_size: Tuple[int, int] = (32, 32), num_classes: int = 10):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self._data = self._generate_mock_data()

    def _generate_mock_data(self) -> List[Tuple[torch.Tensor, int]]:
        """Generate mock data."""
        data = []
        for i in range(self.size):
            # Generate random image
            image = torch.randn(3, *self.image_size)
            # Generate random label
            label = np.random.randint(0, self.num_classes)
            data.append((image, label))
        return data

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._data[idx % len(self._data)]


class MockCIFAR10Dataset(MockDataset):
    """Mock CIFAR-10 dataset."""

    def __init__(self, train: bool = True, download: bool = True, transform=None):
        super().__init__(
            size=50000 if train else 10000,
            image_size=(32, 32),
            num_classes=10
        )
        self.train = train
        self.transform = transform
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self._data[idx % len(self._data)]

        if self.transform:
            image = self.transform(image)

        return image, label


class MockImageNetDataset(MockDataset):
    """Mock ImageNet dataset."""

    def __init__(self, split: str = 'train', download: bool = True, transform=None):
        size = 1281167 if split == 'train' else 50000
        super().__init__(
            size=size,
            image_size=(224, 224),
            num_classes=1000
        )
        self.split = split
        self.transform = transform


class MockDataModule:
    """Mock data module for testing."""

    def __init__(self, dataset_name: str = 'cifar10', batch_size: int = 32, num_workers: int = 0):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._setup_called = False

    def setup(self, stage: str = None):
        """Setup datasets."""
        self._setup_called = True

        if self.dataset_name == 'cifar10':
            self.train_dataset = MockCIFAR10Dataset(train=True)
            self.val_dataset = MockCIFAR10Dataset(train=False)
            self.test_dataset = MockCIFAR10Dataset(train=False)
        elif self.dataset_name == 'imagenet':
            self.train_dataset = MockImageNetDataset(split='train')
            self.val_dataset = MockImageNetDataset(split='val')
            self.test_dataset = MockImageNetDataset(split='val')
        else:
            self.train_dataset = MockDataset()
            self.val_dataset = MockDataset()
            self.test_dataset = MockDataset()

    def train_dataloader(self):
        """Get training dataloader."""
        return MockDataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        """Get validation dataloader."""
        return MockDataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """Get test dataloader."""
        return MockDataLoader(self.test_dataset, batch_size=self.batch_size)


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __iter__(self):
        """Iterate over batches."""
        num_batches = len(self.dataset) // self.batch_size
        for batch_idx in range(num_batches):
            batch_data = []
            batch_labels = []

            for i in range(self.batch_size):
                idx = batch_idx * self.batch_size + i
                data, label = self.dataset[idx]
                batch_data.append(data)
                batch_labels.append(label)

            yield torch.stack(batch_data), torch.tensor(batch_labels)

    def __len__(self):
        """Get number of batches."""
        return len(self.dataset) // self.batch_size


class MockTransform:
    """Mock data transformation."""

    def __init__(self, name: str = 'mock_transform'):
        self.name = name
        self.applied_count = 0

    def __call__(self, x):
        """Apply transformation."""
        self.applied_count += 1
        return x


class MockResizeTransform(MockTransform):
    """Mock resize transformation."""

    def __init__(self, size: Tuple[int, int]):
        super().__init__(f'resize_{size}')
        self.size = size

    def __call__(self, x):
        """Apply resize transformation."""
        super().__call__(x)
        # Mock resize operation
        if isinstance(x, torch.Tensor):
            return torch.nn.functional.interpolate(
                x.unsqueeze(0), size=self.size, mode='bilinear'
            ).squeeze(0)
        return x


class MockNormalizeTransform(MockTransform):
    """Mock normalization transformation."""

    def __init__(self, mean: List[float], std: List[float]):
        super().__init__(f'normalize_{mean}_{std}')
        self.mean = mean
        self.std = std

    def __call__(self, x):
        """Apply normalization."""
        super().__call__(x)
        # Mock normalization
        return x


class MockAugmentationTransform(MockTransform):
    """Mock augmentation transformation."""

    def __init__(self, augmentation_type: str = 'random_flip'):
        super().__init__(f'augment_{augmentation_type}')
        self.augmentation_type = augmentation_type

    def __call__(self, x):
        """Apply augmentation."""
        super().__call__(x)
        # Mock augmentation (just return original)
        return x


@pytest.fixture
def mock_cifar10_dataset():
    """Fixture providing mock CIFAR-10 dataset."""
    return MockCIFAR10Dataset()


@pytest.fixture
def mock_imagenet_dataset():
    """Fixture providing mock ImageNet dataset."""
    return MockImageNetDataset()


@pytest.fixture
def mock_data_module():
    """Fixture providing mock data module."""
    return MockDataModule()


@pytest.fixture
def sample_images():
    """Fixture providing sample images."""
    images = []
    for i in range(5):
        # Create random image tensor
        image = torch.randn(3, 32, 32)
        images.append(image)
    return images


@pytest.fixture
def sample_labels():
    """Fixture providing sample labels."""
    return torch.randint(0, 10, (5,))


@pytest.fixture
def sample_batch_data():
    """Fixture providing sample batch data."""
    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


@pytest.fixture
def mock_transforms():
    """Fixture providing mock transformations."""
    return {
        'resize': MockResizeTransform((224, 224)),
        'normalize': MockNormalizeTransform([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'augment': MockAugmentationTransform('random_flip')
    }


@pytest.fixture
def data_config():
    """Fixture providing data configuration."""
    return {
        'dataset_name': 'cifar10',
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True,
        'image_size': 224,
        'crop_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'train_transforms': [
            'resize',
            'random_crop',
            'random_horizontal_flip',
            'normalize'
        ],
        'val_transforms': [
            'resize',
            'center_crop',
            'normalize'
        ]
    }


@pytest.fixture
def temp_data_dir():
    """Fixture providing temporary data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / 'data'
        data_dir.mkdir(exist_ok=True)

        # Create mock data structure
        (data_dir / 'train').mkdir(exist_ok=True)
        (data_dir / 'val').mkdir(exist_ok=True)
        (data_dir / 'test').mkdir(exist_ok=True)

        yield data_dir


@pytest.fixture
def mock_dataset_factory():
    """Fixture providing dataset factory."""
    def create_dataset(name: str, split: str = 'train', **kwargs):
        """Create mock dataset by name."""
        if name == 'cifar10':
            return MockCIFAR10Dataset(train=(split == 'train'), **kwargs)
        elif name == 'imagenet':
            return MockImageNetDataset(split=split, **kwargs)
        else:
            return MockDataset(**kwargs)

    return create_dataset


@pytest.fixture
def data_loader_config():
    """Fixture providing data loader configuration."""
    return {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'drop_last': True,
        'persistent_workers': True,
        'prefetch_factor': 2
    }


@pytest.fixture
def mock_data_splitter():
    """Fixture providing data splitting utilities."""
    class DataSplitter:
        def __init__(self):
            self.splits = {}

        def split_dataset(self, dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
            """Split dataset into train/val/test."""
            assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

            total_size = len(dataset)
            train_size = int(total_size * train_ratio)
            val_size = int(total_size * val_ratio)
            test_size = total_size - train_size - val_size

            # Mock splitting logic
            indices = list(range(total_size))
            np.random.seed(random_seed)
            np.random.shuffle(indices)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            self.splits = {
                'train': train_indices,
                'val': val_indices,
                'test': test_indices
            }

            return self.splits

        def get_split_datasets(self, dataset, splits):
            """Get split datasets."""
            return {
                split_name: MockSubsetDataset(dataset, indices)
                for split_name, indices in splits.items()
            }

    return DataSplitter()


class MockSubsetDataset:
    """Mock subset dataset."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


@pytest.fixture
def data_validation_utils():
    """Fixture providing data validation utilities."""
    class DataValidator:
        def __init__(self):
            self.validation_errors = []

        def validate_dataset(self, dataset):
            """Validate dataset."""
            self.validation_errors.clear()

            if len(dataset) == 0:
                self.validation_errors.append("Dataset is empty")

            # Try to access first item
            try:
                sample = dataset[0]
                if not isinstance(sample, tuple) or len(sample) != 2:
                    self.validation_errors.append("Dataset items should be (data, label) tuples")
            except Exception as e:
                self.validation_errors.append(f"Cannot access dataset items: {e}")

            return len(self.validation_errors) == 0

        def validate_dataloader(self, dataloader):
            """Validate dataloader."""
            self.validation_errors.clear()

            try:
                # Try to get one batch
                batch = next(iter(dataloader))
                if not isinstance(batch, tuple) or len(batch) != 2:
                    self.validation_errors.append("Batch should be (data, labels) tuple")

                data, labels = batch
                if not isinstance(data, torch.Tensor):
                    self.validation_errors.append("Batch data should be tensor")
                if not isinstance(labels, torch.Tensor):
                    self.validation_errors.append("Batch labels should be tensor")

            except Exception as e:
                self.validation_errors.append(f"Cannot iterate dataloader: {e}")

            return len(self.validation_errors) == 0

        def get_validation_errors(self):
            """Get validation errors."""
            return self.validation_errors.copy()

    return DataValidator()


@pytest.fixture(autouse=True)
def mock_data_dependencies():
    """Fixture to mock data-related dependencies."""
    with patch('torchvision.datasets.CIFAR10', MockCIFAR10Dataset):
        with patch('torchvision.datasets.ImageNet', MockImageNetDataset):
            with patch('torch.utils.data.DataLoader', MockDataLoader):
                yield