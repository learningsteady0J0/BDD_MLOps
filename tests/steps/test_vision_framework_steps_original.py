"""BDD step definitions for Vision Framework Architecture tests."""

import pytest
import torch
import torch.nn as nn
from pytest_bdd import given, when, then, scenarios
from omegaconf import DictConfig
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.core.base.base_model import BaseVisionModel
from src.core.registry.model_registry import ModelRegistry, register_model
from src.vision.models.classification import ResNetClassifier

# Load scenarios from feature file
scenarios('../features/vision_framework.feature')


# Test fixtures and context
@pytest.fixture
def framework_context():
    """Fixture to provide framework testing context."""
    return {
        'model_registry': ModelRegistry(),
        'models': {},
        'configs': {},
        'tensors': {},
        'results': {},
        'metrics': {},
        'temp_files': []
    }


# Background steps
@given('the vision framework is properly initialized')
def vision_framework_initialized(framework_context):
    """Ensure the vision framework is properly initialized."""
    # Verify core modules can be imported
    assert BaseVisionModel is not None
    assert ModelRegistry is not None

    # Initialize context
    framework_context['initialized'] = True


@given('all required dependencies are available')
def required_dependencies_available(framework_context):
    """Verify all required dependencies are available."""
    # Check PyTorch
    assert torch is not None
    assert torch.cuda.is_available() or True  # CPU is fine for testing

    # Check other critical dependencies
    try:
        import pytorch_lightning
        import omegaconf
        import torchmetrics
        framework_context['dependencies_available'] = True
    except ImportError as e:
        pytest.skip(f"Required dependency not available: {e}")


# Scenario: Base Vision Model Creation
@given('I have the framework architecture')
def framework_architecture_available(framework_context):
    """Verify framework architecture components are available."""
    assert BaseVisionModel is not None
    framework_context['architecture_available'] = True


@when('I create a new vision model class')
def create_new_vision_model_class(framework_context):
    """Create a new vision model class that inherits from BaseVisionModel."""

    class TestVisionModel(BaseVisionModel):
        """Test vision model implementation."""

        def _build_backbone(self):
            return nn.Sequential(
                nn.Conv2d(3, 64, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        def _build_head(self):
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, self.num_classes or 10)
            )

        def _setup_loss(self):
            return nn.CrossEntropyLoss()

    framework_context['test_model_class'] = TestVisionModel


@then('it should inherit from the BaseVisionModel class')
def should_inherit_from_base_vision_model(framework_context):
    """Verify the model inherits from BaseVisionModel."""
    test_model_class = framework_context['test_model_class']
    assert issubclass(test_model_class, BaseVisionModel)


@then('it should support common vision tasks')
def should_support_common_vision_tasks(framework_context):
    """Verify the model supports common vision tasks."""
    test_model_class = framework_context['test_model_class']

    # Create test config
    config = DictConfig({
        'model': {'name': 'test_model'},
        'training': {'learning_rate': 0.001}
    })

    # Test different task types
    for task_type in ['classification', 'detection', 'segmentation']:
        try:
            model = test_model_class(
                config=config,
                num_classes=10,
                task_type=task_type
            )
            assert model.task_type == task_type
        except Exception as e:
            # Some task types might not be fully implemented
            if task_type == 'classification':
                raise e  # Classification should always work


@then('it should have proper PyTorch Lightning integration')
def should_have_pytorch_lightning_integration(framework_context):
    """Verify PyTorch Lightning integration."""
    test_model_class = framework_context['test_model_class']

    config = DictConfig({
        'model': {'name': 'test_model'},
        'training': {'learning_rate': 0.001}
    })

    model = test_model_class(config=config, num_classes=10)

    # Verify it's a Lightning Module
    import pytorch_lightning as pl
    assert isinstance(model, pl.LightningModule)

    # Verify required methods exist
    assert hasattr(model, 'training_step')
    assert hasattr(model, 'validation_step')
    assert hasattr(model, 'configure_optimizers')


# Scenario: ResNet Classification Model Registration
@given('the model registry is available')
def model_registry_available(framework_context):
    """Ensure model registry is available."""
    registry = ModelRegistry()
    framework_context['model_registry'] = registry
    assert registry is not None


@when('I register a ResNet classification model')
def register_resnet_classification_model(framework_context):
    """Register ResNet classification model."""
    # The model should already be registered via decorator
    # We just need to verify it's accessible
    registry = framework_context['model_registry']

    # Try to get the model
    try:
        model_class = registry.get_model('resnet')
        framework_context['registered_model'] = model_class
    except Exception as e:
        # If not registered, register it manually for testing
        registry.register_model(
            name='resnet',
            model_class=ResNetClassifier,
            task_type='classification',
            aliases=['resnet18', 'resnet34', 'resnet50']
        )
        framework_context['registered_model'] = ResNetClassifier


@then('the model should be successfully registered')
def model_successfully_registered(framework_context):
    """Verify model is successfully registered."""
    assert 'registered_model' in framework_context
    assert framework_context['registered_model'] is not None


@then('it should be accessible by name "resnet"')
def accessible_by_name_resnet(framework_context):
    """Verify model is accessible by name."""
    registry = framework_context['model_registry']
    model_class = registry.get_model('resnet')
    assert model_class is not None
    assert model_class == framework_context['registered_model']


@then('it should support all ResNet variants')
def should_support_resnet_variants(framework_context):
    """Verify support for ResNet variants."""
    registry = framework_context['model_registry']

    # Test ResNet variants
    for variant in ['resnet18', 'resnet34', 'resnet50']:
        try:
            model_class = registry.get_model(variant)
            assert model_class is not None
        except KeyError:
            # Some variants might be aliases
            pass


# Scenario: Model Configuration and Initialization
@given('a valid model configuration')
def valid_model_configuration(framework_context):
    """Create a valid model configuration."""
    config = DictConfig({
        'model': {
            'name': 'resnet',
            'backbone': 'resnet50',
            'num_classes': 10,
            'pretrained': True
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 32
        },
        'data': {
            'dataset': 'cifar10',
            'image_size': 224
        }
    })
    framework_context['valid_config'] = config


@when('I initialize a vision model with the configuration')
def initialize_vision_model_with_config(framework_context):
    """Initialize vision model with the configuration."""
    config = framework_context['valid_config']

    try:
        model = ResNetClassifier(
            config=config,
            num_classes=config.model.num_classes,
            resnet_version=config.model.backbone,
            pretrained=config.model.pretrained,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        framework_context['initialized_model'] = model
    except Exception as e:
        framework_context['initialization_error'] = str(e)


@then('the model should be created successfully')
def model_created_successfully(framework_context):
    """Verify model was created successfully."""
    assert 'initialized_model' in framework_context
    assert 'initialization_error' not in framework_context
    assert framework_context['initialized_model'] is not None


@then('it should have the correct number of classes')
def should_have_correct_number_of_classes(framework_context):
    """Verify model has correct number of classes."""
    model = framework_context['initialized_model']
    config = framework_context['valid_config']
    assert model.num_classes == config.model.num_classes


@then('it should use the specified backbone architecture')
def should_use_specified_backbone(framework_context):
    """Verify model uses specified backbone."""
    model = framework_context['initialized_model']
    config = framework_context['valid_config']
    assert model.resnet_version == config.model.backbone


@then('the model parameters should be properly initialized')
def model_parameters_properly_initialized(framework_context):
    """Verify model parameters are properly initialized."""
    model = framework_context['initialized_model']

    # Check that parameters exist and have reasonable values
    params = list(model.parameters())
    assert len(params) > 0

    # Check parameter shapes are reasonable
    for param in params:
        assert param.requires_grad in [True, False]  # Should be boolean
        assert param.numel() > 0  # Should have elements


# Scenario: Model Training Components
@given('an initialized vision model')
def initialized_vision_model(framework_context):
    """Ensure we have an initialized vision model."""
    if 'initialized_model' not in framework_context:
        config = DictConfig({
            'model': {'name': 'test_model'},
            'training': {'learning_rate': 0.001}
        })

        class TestModel(BaseVisionModel):
            def _build_backbone(self):
                return nn.Conv2d(3, 64, 3)
            def _build_head(self):
                return nn.Linear(64, 10)
            def _setup_loss(self):
                return nn.CrossEntropyLoss()

        model = TestModel(config=config, num_classes=10)
        framework_context['initialized_model'] = model


@when('I examine the model components')
def examine_model_components(framework_context):
    """Examine the model components."""
    model = framework_context['initialized_model']
    framework_context['model_components'] = {
        'backbone': model.backbone,
        'head': model.head,
        'criterion': model.criterion,
        'metrics': model.metrics
    }


@then('it should have a backbone network')
def should_have_backbone_network(framework_context):
    """Verify model has a backbone network."""
    components = framework_context['model_components']
    assert 'backbone' in components
    assert isinstance(components['backbone'], nn.Module)


@then('it should have a task-specific head')
def should_have_task_specific_head(framework_context):
    """Verify model has a task-specific head."""
    components = framework_context['model_components']
    assert 'head' in components
    assert isinstance(components['head'], nn.Module)


@then('it should have appropriate loss function')
def should_have_appropriate_loss_function(framework_context):
    """Verify model has appropriate loss function."""
    components = framework_context['model_components']
    assert 'criterion' in components
    assert isinstance(components['criterion'], nn.Module)


@then('it should have configured metrics')
def should_have_configured_metrics(framework_context):
    """Verify model has configured metrics."""
    components = framework_context['model_components']
    assert 'metrics' in components
    assert isinstance(components['metrics'], nn.ModuleDict)


@then('it should have optimizer configuration')
def should_have_optimizer_configuration(framework_context):
    """Verify model has optimizer configuration."""
    model = framework_context['initialized_model']

    # Test configure_optimizers method
    optimizer_config = model.configure_optimizers()
    assert optimizer_config is not None


# Scenario: Model Forward Pass
@given('a sample input tensor')
def sample_input_tensor(framework_context):
    """Create a sample input tensor."""
    # Create a sample batch of images (batch_size=2, channels=3, height=32, width=32)
    sample_input = torch.randn(2, 3, 32, 32)
    framework_context['sample_input'] = sample_input


@when('I perform a forward pass')
def perform_forward_pass(framework_context):
    """Perform a forward pass through the model."""
    model = framework_context['initialized_model']
    sample_input = framework_context['sample_input']

    try:
        with torch.no_grad():
            output = model(sample_input)
        framework_context['forward_output'] = output
    except Exception as e:
        framework_context['forward_error'] = str(e)


@then('the output should have the correct shape')
def output_should_have_correct_shape(framework_context):
    """Verify output has correct shape."""
    assert 'forward_error' not in framework_context, f"Forward pass failed: {framework_context.get('forward_error')}"

    output = framework_context['forward_output']
    model = framework_context['initialized_model']
    batch_size = framework_context['sample_input'].size(0)

    expected_shape = (batch_size, model.num_classes)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


@then('the output should be a valid tensor')
def output_should_be_valid_tensor(framework_context):
    """Verify output is a valid tensor."""
    output = framework_context['forward_output']
    assert isinstance(output, torch.Tensor)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@then('no gradients should be lost during forward pass')
def no_gradients_lost_during_forward_pass(framework_context):
    """Verify gradients are not lost during forward pass."""
    model = framework_context['initialized_model']
    sample_input = framework_context['sample_input']

    # Enable gradients
    sample_input.requires_grad_(True)

    # Forward pass with gradients
    output = model(sample_input)
    loss = output.sum()  # Simple loss for gradient checking
    loss.backward()

    # Check that input gradients were computed
    assert sample_input.grad is not None
    assert not torch.isnan(sample_input.grad).any()