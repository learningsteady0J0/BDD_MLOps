"""Logging utilities for the Vision framework."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl


def setup_logger(config: DictConfig) -> logging.Logger:
    """Set up Python logger."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler
    log_dir = Path(config.paths.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def log_hyperparameters(
    logger: Any,
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer
) -> None:
    """Log hyperparameters to experiment tracker."""
    hparams = {}

    # Add config parameters
    hparams.update(OmegaConf.to_container(config, resolve=True))

    # Add model info
    hparams["model/parameters_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/parameters_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # Add data info
    if hasattr(datamodule, "num_classes"):
        hparams["data/num_classes"] = datamodule.num_classes

    # Log to logger
    if hasattr(logger, "log_hyperparams"):
        logger.log_hyperparams(hparams)