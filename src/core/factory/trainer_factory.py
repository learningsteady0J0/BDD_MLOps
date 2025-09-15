"""Factory for creating PyTorch Lightning trainers."""

from typing import List, Optional, Any
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    DeviceStatsMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
import mlflow

from src.core.registry import CallbackRegistry


class TrainerFactory:
    """Factory for creating configured PyTorch Lightning trainers."""

    @staticmethod
    def create_trainer(config: DictConfig) -> pl.Trainer:
        """
        Create a PyTorch Lightning trainer from configuration.

        Args:
            config: Hydra configuration object

        Returns:
            Configured PyTorch Lightning Trainer
        """
        # Create callbacks
        callbacks = TrainerFactory.create_callbacks(config)

        # Create logger
        logger = TrainerFactory.create_logger(config)

        # Get trainer configuration
        trainer_config = OmegaConf.to_container(config.trainer, resolve=True)

        # Remove _target_ if present (used for instantiation)
        trainer_config.pop("_target_", None)

        # Create trainer
        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            **trainer_config
        )

        return trainer

    @staticmethod
    def create_callbacks(config: DictConfig) -> List[pl.callbacks.Callback]:
        """Create training callbacks from configuration."""
        callbacks = []

        # Model Checkpoint
        if config.get("callbacks", {}).get("checkpoint", {}).get("enable", True):
            checkpoint_config = config.callbacks.get("checkpoint", {})
            checkpoint_callback = ModelCheckpoint(
                dirpath=Path(config.paths.output_dir) / "checkpoints",
                filename="{epoch:02d}-{val_loss:.2f}",
                monitor=checkpoint_config.get("monitor", "val/loss"),
                mode=checkpoint_config.get("mode", "min"),
                save_top_k=checkpoint_config.get("save_top_k", 3),
                save_last=checkpoint_config.get("save_last", True),
                verbose=checkpoint_config.get("verbose", True),
                auto_insert_metric_name=False,
            )
            callbacks.append(checkpoint_callback)

        # Early Stopping
        if config.get("callbacks", {}).get("early_stopping", {}).get("enable", False):
            early_stop_config = config.callbacks.get("early_stopping", {})
            early_stop_callback = EarlyStopping(
                monitor=early_stop_config.get("monitor", "val/loss"),
                min_delta=early_stop_config.get("min_delta", 0.001),
                patience=early_stop_config.get("patience", 10),
                mode=early_stop_config.get("mode", "min"),
                verbose=early_stop_config.get("verbose", True),
            )
            callbacks.append(early_stop_callback)

        # Learning Rate Monitor
        if config.get("callbacks", {}).get("lr_monitor", {}).get("enable", True):
            lr_monitor = LearningRateMonitor(
                logging_interval="step",
                log_momentum=True,
            )
            callbacks.append(lr_monitor)

        # Rich Progress Bar
        if config.get("callbacks", {}).get("progress_bar", {}).get("enable", True):
            progress_bar = RichProgressBar(
                refresh_rate=config.callbacks.get("progress_bar", {}).get("refresh_rate", 10),
                leave=config.callbacks.get("progress_bar", {}).get("leave", True),
            )
            callbacks.append(progress_bar)

        # Device Stats Monitor (for debugging)
        if config.get("debug", {}).get("monitor_device", False):
            device_monitor = DeviceStatsMonitor()
            callbacks.append(device_monitor)

        # Stochastic Weight Averaging
        if config.get("callbacks", {}).get("swa", {}).get("enable", False):
            swa_config = config.callbacks.get("swa", {})
            swa_callback = StochasticWeightAveraging(
                swa_lrs=swa_config.get("swa_lrs", 0.05),
                swa_epoch_start=swa_config.get("swa_epoch_start", 0.8),
                annealing_epochs=swa_config.get("annealing_epochs", 10),
                annealing_strategy=swa_config.get("annealing_strategy", "cos"),
            )
            callbacks.append(swa_callback)

        # Custom callbacks from registry
        if config.get("callbacks", {}).get("custom", []):
            for callback_name in config.callbacks.custom:
                callback = CallbackRegistry.create_callback(
                    callback_name,
                    config=config
                )
                callbacks.append(callback)

        # Vision-specific callbacks
        from src.core.base import VisualizationCallback, ModelAnalysisCallback

        if config.get("callbacks", {}).get("visualization", {}).get("enable", False):
            viz_config = config.callbacks.get("visualization", {})
            viz_callback = VisualizationCallback(
                config=config,
                log_dir=Path(config.paths.output_dir) / "visualizations",
                log_every_n_epochs=viz_config.get("log_every_n_epochs", 5),
                num_samples=viz_config.get("num_samples", 8),
            )
            callbacks.append(viz_callback)

        if config.get("callbacks", {}).get("model_analysis", {}).get("enable", False):
            analysis_config = config.callbacks.get("model_analysis", {})
            analysis_callback = ModelAnalysisCallback(
                config=config,
                log_dir=Path(config.paths.output_dir) / "analysis",
                analyze_gradients=analysis_config.get("analyze_gradients", True),
                analyze_weights=analysis_config.get("analyze_weights", True),
                analyze_activations=analysis_config.get("analyze_activations", False),
            )
            callbacks.append(analysis_callback)

        return callbacks

    @staticmethod
    def create_logger(config: DictConfig) -> Optional[Any]:
        """Create experiment logger from configuration."""
        logger_config = config.get("logger", {})

        if not logger_config or not logger_config.get("enable", True):
            return None

        logger_type = logger_config.get("type", "mlflow")

        if logger_type == "mlflow":
            # Set up MLflow
            mlflow.set_tracking_uri(logger_config.get("tracking_uri", "file:./mlruns"))

            logger = MLFlowLogger(
                experiment_name=config.experiment.name,
                run_name=logger_config.get("run_name", None),
                tracking_uri=logger_config.get("tracking_uri", "file:./mlruns"),
                tags=dict(config.experiment.get("tags", {})),
                save_dir=logger_config.get("save_dir", "./mlruns"),
                log_model=logger_config.get("log_model", True),
            )

            # Log hyperparameters
            if logger_config.get("log_hyperparameters", True):
                logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

        elif logger_type == "tensorboard":
            logger = TensorBoardLogger(
                save_dir=Path(config.paths.output_dir) / "tensorboard",
                name=config.experiment.name,
                version=logger_config.get("version", None),
                log_graph=logger_config.get("log_graph", False),
                default_hp_metric=logger_config.get("default_hp_metric", True),
            )

        else:
            raise ValueError(f"Unknown logger type: {logger_type}")

        return logger

    @staticmethod
    def create_profiler(config: DictConfig) -> Optional[Any]:
        """Create profiler from configuration."""
        profiler_type = config.get("debug", {}).get("profiler", None)

        if not profiler_type:
            return None

        if profiler_type == "simple":
            from pytorch_lightning.profilers import SimpleProfiler
            return SimpleProfiler(
                dirpath=Path(config.paths.output_dir) / "profiler",
                filename="simple_profiler",
            )

        elif profiler_type == "advanced":
            from pytorch_lightning.profilers import AdvancedProfiler
            return AdvancedProfiler(
                dirpath=Path(config.paths.output_dir) / "profiler",
                filename="advanced_profiler",
            )

        elif profiler_type == "pytorch":
            from pytorch_lightning.profilers import PyTorchProfiler
            return PyTorchProfiler(
                dirpath=Path(config.paths.output_dir) / "profiler",
                filename="pytorch_profiler",
                group_by_input_shapes=True,
                emit_nvtx=torch.cuda.is_available(),
            )

        else:
            raise ValueError(f"Unknown profiler type: {profiler_type}")

    @staticmethod
    def configure_environment(config: DictConfig) -> None:
        """Configure training environment settings."""
        import os
        import torch

        # Set number of threads
        if "num_threads" in config.get("hardware", {}):
            torch.set_num_threads(config.hardware.num_threads)

        # Set CUDA settings
        if torch.cuda.is_available():
            # Enable cuDNN benchmark for performance
            torch.backends.cudnn.benchmark = config.get("reproducibility", {}).get("benchmark", True)

            # Set deterministic mode if needed
            if config.get("reproducibility", {}).get("deterministic", False):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            # CUDA memory settings
            if config.get("hardware", {}).get("cuda_empty_cache", False):
                torch.cuda.empty_cache()

        # Set environment variables
        env_vars = config.get("environment", {})
        for key, value in env_vars.items():
            os.environ[key] = str(value)