"""Model-related test fixtures for BDD testing."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import DictConfig
from unittest.mock import Mock, patch
import tempfile
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.core.base.base_model import BaseVisionModel
from src.core.registry.model_registry import ModelRegistry


class TestVisionModel(BaseVisionModel):
    """Test vision model for BDD testing."""

    def __init__(self, config: DictConfig, **kwargs):
        """Initialize test vision model."""
        super().__init__(config=config, **kwargs)

    def _build_backbone(self) -> nn.Module:
        """Build simple test backbone."""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def _build_head(self) -> nn.Module:
        """Build simple test head."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, self.num_classes or 10)
        )

    def _setup_loss(self) -> nn.Module:
        """Setup test loss function."""
        return nn.CrossEntropyLoss()


class MockResNetBackbone(nn.Module):
    """Mock ResNet backbone for testing."""

    def __init__(self, feature_dim=2048):
        super().__init__()
        self.feature_dim = feature_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.features(x)


@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return DictConfig({
        'model': {
            'name': 'test_model',
            'backbone': 'resnet50',
            'num_classes': 10,
            'pretrained': False
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 32,
            'epochs': 10
        },
        'data': {
            'dataset': 'cifar10',
            'image_size': 224,
            'num_workers': 4
        },
        'logging': {
            'log_every_n_steps': 50,
            'save_dir': '/tmp/test_logs'
        }
    })


@pytest.fixture
def classification_config():
    """Fixture providing classification model configuration."""
    return DictConfig({
        'model': {
            'name': 'test_classifier',
            'task_type': 'classification',
            'num_classes': 5,
            'backbone': 'resnet18',
            'pretrained': True,
            'freeze_backbone': False
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': {
                'type': 'cosine',
                'T_max': 100
            }
        }
    })


@pytest.fixture
def test_vision_model(test_config):
    """Fixture providing a test vision model instance."""
    return TestVisionModel(
        config=test_config,
        num_classes=test_config.model.num_classes,
        task_type='classification'
    )


@pytest.fixture
def mock_model_registry():
    """Fixture providing a mock model registry."""
    registry = ModelRegistry()

    # Register test models
    registry.register_model(
        name='test_model',
        model_class=TestVisionModel,
        task_type='classification'
    )

    return registry


@pytest.fixture
def sample_batch():
    """Fixture providing sample training batch."""
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)  # Images
    y = torch.randint(0, 10, (batch_size,))  # Labels
    return x, y


@pytest.fixture
def sample_input_tensor():
    """Fixture providing sample input tensor."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def temp_checkpoint_dir():
    """Fixture providing temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_pretrained_weights():
    """Fixture providing mock pretrained weights."""
    weights = {
        'backbone.conv1.weight': torch.randn(64, 3, 7, 7),
        'backbone.conv1.bias': torch.randn(64),
        'head.linear.weight': torch.randn(1000, 2048),
        'head.linear.bias': torch.randn(1000)
    }
    return weights


@pytest.fixture
def model_factory():
    """Fixture providing a model factory function."""
    def _create_model(model_type='classification', num_classes=10, **kwargs):
        config = DictConfig({
            'model': {'name': f'test_{model_type}'},
            'training': {'learning_rate': 0.001}
        })

        return TestVisionModel(
            config=config,
            num_classes=num_classes,
            task_type=model_type,
            **kwargs
        )

    return _create_model


@pytest.fixture
def metric_tracker():
    """Fixture providing metric tracking utilities."""
    class MetricTracker:
        def __init__(self):
            self.metrics = {}
            self.history = []

        def update(self, metrics_dict):
            self.metrics.update(metrics_dict)
            self.history.append(metrics_dict.copy())

        def get_latest(self):
            return self.metrics

        def get_history(self):
            return self.history

        def reset(self):
            self.metrics.clear()
            self.history.clear()

    return MetricTracker()


@pytest.fixture
def mock_lightning_trainer():
    """Fixture providing mock PyTorch Lightning trainer."""
    trainer = Mock()
    trainer.fit = Mock()
    trainer.test = Mock()
    trainer.validate = Mock()
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.is_global_zero = True

    return trainer


@pytest.fixture
def training_callbacks():
    """Fixture providing training callbacks."""
    callbacks = []

    # Mock ModelCheckpoint
    checkpoint_callback = Mock()
    checkpoint_callback.save_checkpoint = Mock()
    checkpoint_callback.best_model_path = '/tmp/best_model.ckpt'
    callbacks.append(checkpoint_callback)

    # Mock EarlyStopping
    early_stopping_callback = Mock()
    early_stopping_callback.should_stop = False
    callbacks.append(early_stopping_callback)

    # Mock LearningRateMonitor
    lr_monitor_callback = Mock()
    lr_monitor_callback.log_lr = Mock()
    callbacks.append(lr_monitor_callback)

    return callbacks


@pytest.fixture
def experiment_config():
    """Fixture providing experiment configuration."""
    return DictConfig({
        'experiment': {
            'name': 'test_experiment',
            'version': '1.0',
            'tags': ['test', 'bdd'],
            'description': 'BDD test experiment'
        },
        'model': {
            'name': 'resnet50',
            'task_type': 'classification',
            'num_classes': 10,
            'pretrained': True
        },
        'data': {
            'name': 'cifar10',
            'batch_size': 32,
            'num_workers': 4,
            'val_split': 0.2
        },
        'training': {
            'max_epochs': 5,
            'learning_rate': 0.001,
            'optimizer': 'adamw',
            'scheduler': 'cosine'
        },
        'logging': {
            'experiment_tracking': 'mlflow',
            'log_model': True,
            'log_every_n_steps': 10
        }
    })


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Fixture to setup test environment."""
    # Set environment variables for testing
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # Force CPU for testing
    monkeypatch.setenv("PYTHONPATH", os.path.join(os.path.dirname(__file__), '../../'))

    # Mock external dependencies that might not be available
    with patch('torch.cuda.is_available', return_value=False):
        with patch('mlflow.start_run'):
            with patch('wandb.init'):
                yield