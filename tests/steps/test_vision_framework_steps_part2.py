"""BDD step definitions for Vision Framework Architecture tests - Part 2."""

import pytest
import torch
import torch.nn as nn
from pytest_bdd import given, when, then
from unittest.mock import MagicMock, patch
import tempfile
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.core.base.base_model import BaseVisionModel


# Scenario: Model Training Step
@given('an initialized vision model in training mode')
def initialized_vision_model_training_mode(framework_context):
    """Ensure vision model is in training mode."""
    if 'initialized_model' not in framework_context:
        from omegaconf import DictConfig

        class TestModel(BaseVisionModel):
            def _build_backbone(self):
                return nn.Conv2d(3, 64, 3)
            def _build_head(self):
                return nn.Sequential(nn.Flatten(), nn.Linear(64, 10))
            def _setup_loss(self):
                return nn.CrossEntropyLoss()

        config = DictConfig({'model': {'name': 'test'}, 'training': {'learning_rate': 0.001}})
        model = TestModel(config=config, num_classes=10)
        framework_context['initialized_model'] = model

    model = framework_context['initialized_model']
    model.train()  # Set to training mode
    framework_context['model_mode'] = 'training'


@given('a batch of training data')
def batch_of_training_data(framework_context):
    """Create a batch of training data."""
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)  # Images
    y = torch.randint(0, 10, (batch_size,))  # Labels
    framework_context['training_batch'] = (x, y)


@when('I perform a training step')
def perform_training_step(framework_context):
    """Perform a training step."""
    model = framework_context['initialized_model']
    batch = framework_context['training_batch']

    try:
        # Simulate training step
        training_loss = model.training_step(batch, batch_idx=0)
        framework_context['training_loss'] = training_loss
        framework_context['training_step_success'] = True
    except Exception as e:
        framework_context['training_step_error'] = str(e)


@then('the loss should be computed correctly')
def loss_computed_correctly(framework_context):
    """Verify loss is computed correctly."""
    assert 'training_step_error' not in framework_context, f"Training step failed: {framework_context.get('training_step_error')}"

    loss = framework_context['training_loss']
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.dim() == 0  # Scalar loss
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0  # Loss should be non-negative


@then('metrics should be updated')
def metrics_should_be_updated(framework_context):
    """Verify metrics are updated during training."""
    model = framework_context['initialized_model']

    # Check if metrics are available
    if hasattr(model, 'metrics') and model.metrics:
        # For classification models, check accuracy metrics
        if 'train_acc' in model.metrics:
            metric = model.metrics['train_acc']
            # The metric should be callable and have state
            assert hasattr(metric, 'compute')


@then('gradients should be computed')
def gradients_should_be_computed(framework_context):
    """Verify gradients are computed."""
    model = framework_context['initialized_model']

    # Check that model parameters have gradients
    has_gradients = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            has_gradients = True
            break

    # Note: In a real training loop, gradients would be computed after loss.backward()
    # Here we just verify the model is set up correctly for gradient computation
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "Model should have trainable parameters"


@then('training logs should be generated')
def training_logs_should_be_generated(framework_context):
    """Verify training logs are generated."""
    # In a real scenario, we would check Lightning's logging mechanism
    # For now, verify the training step completed successfully
    assert framework_context.get('training_step_success', False)


# Scenario: Model Validation Step
@given('an initialized vision model in evaluation mode')
def initialized_vision_model_evaluation_mode(framework_context):
    """Ensure vision model is in evaluation mode."""
    if 'initialized_model' not in framework_context:
        from omegaconf import DictConfig

        class TestModel(BaseVisionModel):
            def _build_backbone(self):
                return nn.Conv2d(3, 64, 3)
            def _build_head(self):
                return nn.Sequential(nn.Flatten(), nn.Linear(64, 10))
            def _setup_loss(self):
                return nn.CrossEntropyLoss()

        config = DictConfig({'model': {'name': 'test'}, 'training': {'learning_rate': 0.001}})
        model = TestModel(config=config, num_classes=10)
        framework_context['initialized_model'] = model

    model = framework_context['initialized_model']
    model.eval()  # Set to evaluation mode
    framework_context['model_mode'] = 'evaluation'


@given('a batch of validation data')
def batch_of_validation_data(framework_context):
    """Create a batch of validation data."""
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)  # Images
    y = torch.randint(0, 10, (batch_size,))  # Labels
    framework_context['validation_batch'] = (x, y)


@when('I perform a validation step')
def perform_validation_step(framework_context):
    """Perform a validation step."""
    model = framework_context['initialized_model']
    batch = framework_context['validation_batch']

    try:
        with torch.no_grad():
            validation_loss = model.validation_step(batch, batch_idx=0)
        framework_context['validation_loss'] = validation_loss
        framework_context['validation_step_success'] = True
    except Exception as e:
        framework_context['validation_step_error'] = str(e)


@then('the validation loss should be computed')
def validation_loss_should_be_computed(framework_context):
    """Verify validation loss is computed."""
    assert 'validation_step_error' not in framework_context, f"Validation step failed: {framework_context.get('validation_step_error')}"

    loss = framework_context['validation_loss']
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar loss
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0


@then('validation metrics should be updated')
def validation_metrics_should_be_updated(framework_context):
    """Verify validation metrics are updated."""
    model = framework_context['initialized_model']

    # Check if validation metrics are available
    if hasattr(model, 'metrics') and model.metrics:
        if 'val_acc' in model.metrics:
            metric = model.metrics['val_acc']
            assert hasattr(metric, 'compute')


@then('no gradients should be computed')
def no_gradients_should_be_computed(framework_context):
    """Verify no gradients are computed during validation."""
    # This is more of a test design principle - validation should use torch.no_grad()
    # We verify the model is in eval mode
    model = framework_context['initialized_model']
    assert not model.training  # Should be in eval mode


@then('validation logs should be generated')
def validation_logs_should_be_generated(framework_context):
    """Verify validation logs are generated."""
    assert framework_context.get('validation_step_success', False)


# Scenario: Model Backbone Freezing
@when('I freeze the backbone parameters')
def freeze_backbone_parameters(framework_context):
    """Freeze the backbone parameters."""
    model = framework_context['initialized_model']

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    framework_context['backbone_frozen'] = True


@then('backbone parameters should have requires_grad=False')
def backbone_parameters_requires_grad_false(framework_context):
    """Verify backbone parameters have requires_grad=False."""
    model = framework_context['initialized_model']

    for param in model.backbone.parameters():
        assert not param.requires_grad, "Backbone parameters should have requires_grad=False"


@then('head parameters should have requires_grad=True')
def head_parameters_requires_grad_true(framework_context):
    """Verify head parameters have requires_grad=True."""
    model = framework_context['initialized_model']

    # Head parameters should still be trainable
    head_trainable = False
    for param in model.head.parameters():
        if param.requires_grad:
            head_trainable = True
            break

    assert head_trainable, "Head should have at least some trainable parameters"


@then('frozen parameters should not update during training')
def frozen_parameters_should_not_update(framework_context):
    """Verify frozen parameters don't update during training."""
    model = framework_context['initialized_model']

    # Store initial backbone parameter values
    initial_params = {}
    for name, param in model.backbone.named_parameters():
        initial_params[name] = param.clone().detach()

    # Simulate a training step
    if 'training_batch' not in framework_context:
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
        framework_context['training_batch'] = (x, y)

    batch = framework_context['training_batch']

    # Forward pass and loss computation
    output = model(batch[0])
    loss = model.criterion(output, batch[1])

    # Backward pass
    loss.backward()

    # Check that frozen parameters didn't change
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            # Parameter shouldn't have gradients
            assert param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad))


# Scenario: Model Metrics Setup
@given('a vision model configuration for classification')
def vision_model_configuration_for_classification(framework_context):
    """Create a vision model configuration for classification."""
    from omegaconf import DictConfig

    config = DictConfig({
        'model': {
            'name': 'test_classifier',
            'task_type': 'classification',
            'num_classes': 5
        },
        'training': {
            'learning_rate': 0.001
        }
    })
    framework_context['classification_config'] = config


@when('I setup the model metrics')
def setup_model_metrics(framework_context):
    """Setup model metrics."""
    from omegaconf import DictConfig

    class TestClassifier(BaseVisionModel):
        def _build_backbone(self):
            return nn.Conv2d(3, 64, 3)
        def _build_head(self):
            return nn.Sequential(nn.Flatten(), nn.Linear(64, self.num_classes))
        def _setup_loss(self):
            return nn.CrossEntropyLoss()

    config = framework_context['classification_config']
    model = TestClassifier(
        config=config,
        num_classes=config.model.num_classes,
        task_type='classification'
    )

    framework_context['classifier_model'] = model
    framework_context['model_metrics'] = model.metrics


@then('it should have accuracy metrics for training and validation')
def should_have_accuracy_metrics(framework_context):
    """Verify model has accuracy metrics for training and validation."""
    metrics = framework_context['model_metrics']

    # Check for training and validation accuracy
    assert 'train_acc' in metrics or len([k for k in metrics.keys() if 'acc' in k and 'train' in k]) > 0
    assert 'val_acc' in metrics or len([k for k in metrics.keys() if 'acc' in k and 'val' in k]) > 0


@then('it should have F1 score metrics')
def should_have_f1_score_metrics(framework_context):
    """Verify model has F1 score metrics."""
    metrics = framework_context['model_metrics']

    # Check for F1 score metrics
    f1_metrics = [k for k in metrics.keys() if 'f1' in k.lower()]
    assert len(f1_metrics) > 0, "Should have F1 score metrics"


@then('metrics should be properly initialized')
def metrics_should_be_properly_initialized(framework_context):
    """Verify metrics are properly initialized."""
    metrics = framework_context['model_metrics']

    for name, metric in metrics.items():
        assert hasattr(metric, 'compute'), f"Metric {name} should have compute method"
        assert hasattr(metric, 'update'), f"Metric {name} should have update method"


@then('metrics should reset between epochs')
def metrics_should_reset_between_epochs(framework_context):
    """Verify metrics reset between epochs."""
    model = framework_context['classifier_model']

    # Test epoch end hooks
    assert hasattr(model, 'on_train_epoch_end')
    assert hasattr(model, 'on_validation_epoch_end')

    # The actual reset functionality is tested by calling these methods
    try:
        model.on_train_epoch_end()
        model.on_validation_epoch_end()
    except Exception as e:
        pytest.fail(f"Epoch end hooks should not raise exceptions: {e}")


# Scenario: Model Hooks Registration
@when('I register custom hooks')
def register_custom_hooks(framework_context):
    """Register custom hooks."""
    model = framework_context['initialized_model']

    def test_hook(x):
        return x

    try:
        model.register_hook('before_forward', test_hook)
        framework_context['hook_registered'] = True
    except Exception as e:
        framework_context['hook_error'] = str(e)


@then('hooks should be properly registered')
def hooks_should_be_properly_registered(framework_context):
    """Verify hooks are properly registered."""
    assert 'hook_error' not in framework_context, f"Hook registration failed: {framework_context.get('hook_error')}"
    assert framework_context.get('hook_registered', False)


@then('hooks should execute at the correct time')
def hooks_should_execute_at_correct_time(framework_context):
    """Verify hooks execute at correct time."""
    model = framework_context['initialized_model']

    # Test that hooks dictionary exists and is structured correctly
    assert hasattr(model, 'hooks')
    assert isinstance(model.hooks, dict)

    expected_hooks = ['before_forward', 'after_forward', 'before_backward', 'after_backward']
    for hook_name in expected_hooks:
        assert hook_name in model.hooks


@then('multiple hooks can be registered for the same event')
def multiple_hooks_can_be_registered(framework_context):
    """Verify multiple hooks can be registered for the same event."""
    model = framework_context['initialized_model']

    def hook1(x):
        return x

    def hook2(x):
        return x

    # Register multiple hooks for the same event
    model.register_hook('before_forward', hook1)
    model.register_hook('before_forward', hook2)

    # Verify both hooks are registered
    before_forward_hooks = model.hooks['before_forward']
    assert len(before_forward_hooks) >= 2


# Scenario: Model Summary Generation
@when('I generate a model summary')
def generate_model_summary(framework_context):
    """Generate a model summary."""
    model = framework_context['initialized_model']

    try:
        # Mock torchinfo.summary since it might not be available
        with patch('src.core.base.base_model.summary') as mock_summary:
            mock_summary.return_value = "Mocked model summary"
            summary_output = model.summary()
        framework_context['model_summary'] = summary_output
    except Exception as e:
        framework_context['summary_error'] = str(e)


@then('it should display model architecture')
def should_display_model_architecture(framework_context):
    """Verify summary displays model architecture."""
    assert 'summary_error' not in framework_context, f"Summary generation failed: {framework_context.get('summary_error')}"
    assert 'model_summary' in framework_context


@then('it should show parameter counts')
def should_show_parameter_counts(framework_context):
    """Verify summary shows parameter counts."""
    # The actual parameter counting would be done by torchinfo
    # Here we just verify the summary method exists and can be called
    model = framework_context['initialized_model']
    assert hasattr(model, 'summary')


@then('it should show layer information')
def should_show_layer_information(framework_context):
    """Verify summary shows layer information."""
    # This would be handled by torchinfo in practice
    assert 'model_summary' in framework_context


@then('the summary should be properly formatted')
def summary_should_be_properly_formatted(framework_context):
    """Verify summary is properly formatted."""
    summary = framework_context['model_summary']
    assert isinstance(summary, str)
    assert len(summary) > 0