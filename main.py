"""Main training script for Vision PyTorch Lightning Framework."""

import os
import sys
from pathlib import Path
from typing import Optional, List

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.core.factory import TrainerFactory
from src.core.registry import ModelRegistry, DataModuleRegistry
from src.utils.logging_utils import setup_logger, log_hyperparameters


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


def setup_environment(config: DictConfig) -> None:
    """Set up the training environment."""
    # Set seed for reproducibility
    if config.get("experiment", {}).get("seed"):
        seed_everything(config.experiment.seed, workers=True)

    # Configure hardware settings
    TrainerFactory.configure_environment(config)

    # Set up logging
    setup_logger(config)


def instantiate_datamodule(config: DictConfig) -> pl.LightningDataModule:
    """Instantiate data module from configuration."""
    data_config = config.data

    # Get datamodule class from registry
    if hasattr(data_config, "_target_"):
        # Direct instantiation path
        module_path = data_config._target_
        module_parts = module_path.split(".")
        module_name = module_parts[-1]
    else:
        # Use registry
        module_name = data_config.name

    # Create datamodule
    datamodule = DataModuleRegistry.create_datamodule(
        name=module_name,
        config=config,
        **OmegaConf.to_container(data_config, resolve=True)
    )

    return datamodule


def instantiate_model(config: DictConfig) -> pl.LightningModule:
    """Instantiate model from configuration."""
    model_config = config.model

    # Get model class from registry
    if hasattr(model_config, "_target_"):
        # Direct instantiation path
        module_path = model_config._target_
        module_parts = module_path.split(".")
        module_name = module_parts[-1]
    else:
        # Use registry
        module_name = model_config.name

    # Create model
    model = ModelRegistry.create_model(
        name=module_name,
        config=config,
        **OmegaConf.to_container(model_config, resolve=True)
    )

    return model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> Optional[float]:
    """
    Main training function.

    Args:
        config: Hydra configuration object

    Returns:
        Best validation metric value
    """
    # Print configuration
    print(OmegaConf.to_yaml(config))

    # Set up environment
    setup_environment(config)

    # Log start of training
    print("\n" + "=" * 80)
    print(f"Starting experiment: {config.experiment.name}")
    print("=" * 80 + "\n")

    # Create data module
    print("Setting up data module...")
    datamodule = instantiate_datamodule(config)
    datamodule.prepare_data()
    datamodule.setup()

    # Log dataset statistics
    dataset_stats = datamodule.get_dataset_stats()
    print(f"Dataset: {config.data.name}")
    print(f"  - Train samples: {dataset_stats['num_train_samples']}")
    print(f"  - Val samples: {dataset_stats['num_val_samples']}")
    print(f"  - Test samples: {dataset_stats['num_test_samples']}")
    print(f"  - Number of classes: {dataset_stats['num_classes']}")
    print(f"  - Input size: {dataset_stats['image_size']}")

    # Create model
    print("\nInitializing model...")
    model = instantiate_model(config)

    # Print model summary
    print(f"Model: {config.model.name}")
    if hasattr(model, "resnet_version"):
        print(f"  - Architecture: {model.resnet_version}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create trainer
    print("\nSetting up trainer...")
    trainer = TrainerFactory.create_trainer(config)

    # Log hyperparameters
    if trainer.logger:
        log_hyperparameters(
            logger=trainer.logger,
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer
        )

    # Training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    # Check if resuming from checkpoint
    ckpt_path = config.get("checkpoint", {}).get("resume_from", None)

    # Fit the model
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path
    )

    # Get best checkpoint path
    best_model_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") else None

    # Testing
    if config.get("test_after_training", True) and best_model_path:
        print("\n" + "=" * 80)
        print("Running test evaluation...")
        print("=" * 80 + "\n")

        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=best_model_path
        )

    # Get best validation metric
    best_metric = None
    if hasattr(trainer, "checkpoint_callback") and trainer.checkpoint_callback:
        best_metric = trainer.checkpoint_callback.best_model_score

    # Save final model
    if config.get("save_final_model", True):
        final_model_path = Path(config.paths.output_dir) / "final_model.ckpt"
        trainer.save_checkpoint(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")

    # Log completion
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    if best_metric is not None:
        print(f"Best validation metric: {best_metric:.4f}")
    print("=" * 80 + "\n")

    return best_metric


if __name__ == "__main__":
    main()