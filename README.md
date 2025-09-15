# PyTorch Lightning AI Training Framework

A comprehensive, plugin-based AI training framework built on PyTorch Lightning with full MLOps integration.

## Architecture Overview

This framework provides a modular, extensible architecture for training AI models across multiple domains (Vision, NLP, Time Series, Audio) with enterprise-grade MLOps capabilities.

### Key Features

- **Plugin Architecture**: Dynamically discover and load models, datasets, and components
- **Multi-Domain Support**: Built-in support for Vision, NLP, Time Series, and Audio
- **MLOps Integration**: MLflow tracking with MongoDB backend
- **Distributed Training**: Support for DDP, FSDP, and DeepSpeed
- **Configuration Management**: Hydra-based configuration system
- **Type Safety**: Comprehensive type hints and contracts
- **Extensibility**: Easy to add new models, datasets, and components

## Project Structure

```
BDD_MLOps/
├── src/
│   ├── core/                  # Core framework components
│   │   ├── base/              # Base classes for models, data, etc.
│   │   ├── registry/          # Plugin registration system
│   │   ├── contracts/         # Interfaces and contracts
│   │   └── utils/             # Utility functions
│   ├── models/                # Model implementations
│   ├── data/                  # Dataset implementations
│   ├── training/              # Training orchestration
│   ├── mlops/                 # MLOps integrations
│   │   ├── tracking/          # Experiment tracking
│   │   ├── storage/           # Model and data storage
│   │   └── deployment/        # Model deployment
│   ├── configs/               # Hydra configuration files
│   └── plugins/               # Domain-specific plugins
│       ├── vision/
│       ├── nlp/
│       └── timeseries/
├── docs/
│   └── architecture/          # Architecture documentation
├── tests/                     # Test files
└── main.py                    # Main entry point
```

## Core Components

### 1. Base Classes

The framework provides abstract base classes that all components must inherit from:

- **BaseModel**: Abstract base for all models (PyTorch Lightning Module)
- **BaseDataModule**: Abstract base for all data modules
- **BaseVisionModel**: Specialized base for vision models
- **BaseNLPModel**: Specialized base for NLP models
- **BaseTimeSeriesModel**: Specialized base for time series models

### 2. Plugin Registry

Dynamic plugin discovery and registration system:

```python
from src.core.registry.plugin_registry import registry, register_model

@register_model("my_model")
class MyModel(BaseModel):
    # Model implementation
    pass

# Use the model
model = registry.create_model("my_model", **config)
```

### 3. Contract System

Interface definitions and validation:

- **ITrainable**: Protocol for trainable components
- **ISerializable**: Protocol for serializable components
- **IConfigurable**: Protocol for configurable components
- **ModelContract**: Contract validation for models
- **DataContract**: Contract validation for data modules

### 4. MLOps Integration

Comprehensive MLflow integration with MongoDB backend:

- Experiment tracking
- Model versioning
- Artifact management
- Metric logging
- Distributed tracking

## Configuration System

The framework uses Hydra for configuration management:

```yaml
# config.yaml
defaults:
  - model: resnet50
  - dataset: imagenet
  - trainer: default

experiment:
  name: ${model.name}_${dataset.name}

mlops:
  tracking:
    backend: mlflow
    uri: mongodb://localhost:27017
```

## Usage

### Basic Training

```python
import hydra
from omegaconf import DictConfig
from src.training.trainer import FrameworkTrainer

@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    trainer = FrameworkTrainer(cfg)
    results = trainer.train()
    return results

if __name__ == "__main__":
    train()
```

### Command Line Interface

```bash
# Train with default configuration
python main.py

# Override configuration
python main.py model=vit dataset=cifar10 trainer.max_epochs=100

# Use different experiment
python main.py +experiment=custom_experiment

# Distributed training
python main.py distributed.enabled=true distributed.devices=4
```

### Creating Custom Models

```python
from src.core.base.model import BaseVisionModel
from src.core.registry.plugin_registry import register_model

@register_model("custom_resnet")
class CustomResNet(BaseVisionModel):
    def _build_model(self):
        # Build your model architecture
        return model

    def _setup_metrics(self):
        # Setup metrics
        return metrics

    def forward(self, x):
        # Forward pass
        return self.model(x)
```

### Creating Custom Datasets

```python
from src.core.base.dataset import BaseVisionDataModule
from src.core.registry.plugin_registry import register_dataset

@register_dataset("custom_dataset")
class CustomDataset(BaseVisionDataModule):
    def prepare_data(self):
        # Download/prepare data
        pass

    def setup(self, stage=None):
        # Setup datasets
        pass
```

## Extensibility

### Adding New Domains

1. Create domain-specific base classes
2. Implement domain plugins
3. Register with the plugin system
4. Add configuration files

### Adding New Components

1. Inherit from appropriate base class
2. Implement required methods
3. Register with plugin registry
4. Add configuration

## MLOps Features

### Experiment Tracking

- Automatic parameter logging
- Metric tracking
- Artifact management
- Model versioning

### Model Registry

- Model registration
- Version management
- Stage transitions (dev → staging → production)
- Model serving preparation

### Distributed Training

- Data Parallel (DP)
- Distributed Data Parallel (DDP)
- Fully Sharded Data Parallel (FSDP)
- DeepSpeed integration

## Performance Optimization

- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Gradient clipping
- Learning rate scheduling
- Early stopping
- Model checkpointing

## Monitoring

- Real-time training metrics
- Resource utilization tracking
- Experiment comparison
- Model performance analysis

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- MLflow
- MongoDB (for MLflow backend)
- Hydra

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BDD_MLOps.git
cd BDD_MLOps

# Install dependencies
pip install -r requirements.txt

# Setup MongoDB (for MLflow backend)
docker run -d -p 27017:27017 mongo

# Start MLflow server
mlflow server --backend-store-uri mongodb://localhost:27017 --default-artifact-root ./mlruns
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

## Contributing

1. Follow the established architecture patterns
2. Implement required interfaces and contracts
3. Add comprehensive type hints
4. Include unit tests
5. Update documentation

## License

MIT License

## Architecture Decisions

### Why PyTorch Lightning?
- Production-ready training loops
- Built-in distributed training support
- Extensive callback system
- Clean separation of concerns

### Why Plugin Architecture?
- Easy extensibility
- Domain isolation
- Dynamic component loading
- Modular development

### Why Hydra?
- Hierarchical configuration
- Command-line overrides
- Configuration composition
- Experiment management

### Why MLflow + MongoDB?
- Scalable experiment tracking
- NoSQL flexibility
- Distributed team support
- Production-ready setup