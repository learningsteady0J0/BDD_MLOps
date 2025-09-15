"""Main entry point for the PyTorch Lightning AI Training Framework."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.training.trainer import FrameworkTrainer
from src.core.registry.plugin_registry import registry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment(cfg: DictConfig) -> None:
    """
    Setup environment variables and global settings.

    Args:
        cfg: Configuration object
    """
    # Set random seeds for reproducibility
    if cfg.reproducibility.seed is not None:
        pl.seed_everything(cfg.reproducibility.seed, workers=True)
        logger.info(f"Set random seed to {cfg.reproducibility.seed}")

    # Set number of threads
    if cfg.resources.cpu.num_threads:
        torch.set_num_threads(cfg.resources.cpu.num_threads)
        os.environ['OMP_NUM_THREADS'] = str(cfg.resources.cpu.num_threads)

    # Configure GPU memory if available
    if torch.cuda.is_available() and cfg.resources.gpu.enabled:
        if cfg.resources.gpu.memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(cfg.resources.gpu.memory_fraction)

    # Set PyTorch settings
    torch.backends.cudnn.benchmark = cfg.reproducibility.benchmark
    torch.backends.cudnn.deterministic = cfg.reproducibility.deterministic


def discover_plugins(cfg: DictConfig) -> None:
    """
    Discover and load plugins.

    Args:
        cfg: Configuration object
    """
    if not cfg.plugins.auto_discover:
        return

    for plugin_dir in cfg.plugins.plugin_dirs:
        plugin_path = Path(plugin_dir)
        if plugin_path.exists():
            logger.info(f"Discovering plugins in {plugin_path}")
            registry.discover_plugins(plugin_path)

    # Log discovered plugins
    logger.info(f"Discovered models: {registry.list_models()}")
    logger.info(f"Discovered datasets: {registry.list_datasets()}")


@hydra.main(version_base=None, config_path="src/configs", config_name="config")
def main(cfg: DictConfig) -> Dict[str, Any]:
    """
    Main training function.

    Args:
        cfg: Hydra configuration object

    Returns:
        Training results
    """
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Setup environment
    setup_environment(cfg)

    # Discover plugins
    discover_plugins(cfg)

    # Create trainer
    trainer = FrameworkTrainer(cfg)

    # Run training
    results = trainer.train()

    # Log results
    if results:
        logger.info("Training Results:")
        for key, value in results.get('train_metrics', {}).items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            logger.info(f"  {key}: {value}")

        if results.get('test_metrics'):
            logger.info("Test Results:")
            for key, value in results['test_metrics'].items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                logger.info(f"  {key}: {value}")

        if results.get('best_model_path'):
            logger.info(f"Best model saved at: {results['best_model_path']}")

    return results


if __name__ == "__main__":
    main()