"""Enhanced trainer with plugin support and MLOps integration."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary
)
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
import torch
from omegaconf import DictConfig, OmegaConf
import logging

from src.core.registry.plugin_registry import registry
from src.mlops.tracking.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


class FrameworkTrainer:
    """
    Enhanced trainer that integrates PyTorch Lightning with the plugin system
    and MLOps capabilities.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the framework trainer.

        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.mlflow_tracker = None
        self.trainer = None
        self.model = None
        self.datamodule = None

        # Initialize components
        self._setup_mlflow()
        self._setup_callbacks()
        self._setup_loggers()
        self._setup_trainer()

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        if self.config.mlops.tracking.backend == "mlflow":
            self.mlflow_tracker = MLflowTracker(
                tracking_uri=self.config.mlops.tracking.uri,
                default_experiment=self.config.mlops.tracking.experiment_name,
                artifact_location=self.config.mlops.tracking.artifact_location
            )
            logger.info("MLflow tracking initialized")

    def _setup_callbacks(self) -> List[pl.Callback]:
        """
        Setup training callbacks.

        Returns:
            List of callbacks
        """
        callbacks = []

        # Model checkpoint callback
        if self.config.checkpoint.enabled:
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.config.checkpoint.dir,
                filename=self.config.checkpoint.filename,
                monitor=self.config.checkpoint.monitor,
                mode=self.config.checkpoint.mode,
                save_top_k=self.config.checkpoint.save_top_k,
                save_last=self.config.checkpoint.save_last,
                save_weights_only=self.config.checkpoint.save_weights_only,
                auto_insert_metric_name=self.config.checkpoint.auto_insert_metric_name,
                verbose=True
            )
            callbacks.append(checkpoint_callback)

        # Early stopping callback
        if self.config.early_stopping.enabled:
            early_stopping = EarlyStopping(
                monitor=self.config.early_stopping.monitor,
                mode=self.config.early_stopping.mode,
                patience=self.config.early_stopping.patience,
                min_delta=self.config.early_stopping.min_delta,
                verbose=self.config.early_stopping.verbose
            )
            callbacks.append(early_stopping)

        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval='step'))

        # Progress bar
        callbacks.append(RichProgressBar(leave=True))

        # Model summary
        callbacks.append(RichModelSummary(max_depth=2))

        # Load additional callbacks from registry
        if 'callbacks' in self.config:
            for callback_name in self.config.callbacks.get('additional', []):
                try:
                    callback_class = registry.get_callback(callback_name)
                    callbacks.append(callback_class())
                    logger.info(f"Loaded callback: {callback_name}")
                except KeyError:
                    logger.warning(f"Callback not found: {callback_name}")

        self.callbacks = callbacks
        return callbacks

    def _setup_loggers(self) -> List[pl.loggers.Logger]:
        """
        Setup experiment loggers.

        Returns:
            List of loggers
        """
        loggers = []

        # MLflow logger
        if self.config.logging.mlflow.enabled:
            mlflow_logger = MLFlowLogger(
                experiment_name=self.config.mlops.tracking.experiment_name,
                run_name=self.config.mlops.tracking.run_name,
                tracking_uri=self.config.mlops.tracking.uri,
                tags=dict(self.config.experiment.tags) if self.config.experiment.tags else None,
                save_dir=self.config.mlops.tracking.artifact_location,
                log_model=True
            )
            loggers.append(mlflow_logger)

        # TensorBoard logger
        if self.config.logging.tensorboard.enabled:
            tensorboard_logger = TensorBoardLogger(
                save_dir=self.config.logging.tensorboard.log_dir,
                name=self.config.experiment.name,
                version=None,
                log_graph=True,
                default_hp_metric=False
            )
            loggers.append(tensorboard_logger)

        self.loggers = loggers
        return loggers

    def _setup_trainer(self) -> None:
        """Setup PyTorch Lightning trainer."""
        # Determine devices and accelerator
        accelerator = self._get_accelerator()
        devices = self._get_devices()
        strategy = self._get_strategy()

        # Create trainer
        self.trainer = Trainer(
            # Training configuration
            max_epochs=self.config.trainer.max_epochs,
            min_epochs=self.config.trainer.get('min_epochs', 1),
            max_steps=self.config.trainer.get('max_steps', -1),

            # Validation configuration
            val_check_interval=self.config.validation.interval,
            check_val_every_n_epoch=self.config.validation.get('check_every_n_epoch', 1),
            num_sanity_val_steps=self.config.validation.sanity_checks,

            # Hardware configuration
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=self.config.distributed.precision,

            # Callbacks and logging
            callbacks=self.callbacks,
            logger=self.loggers,
            log_every_n_steps=self.config.logging.mlflow.log_every_n_steps,

            # Performance
            enable_checkpointing=self.config.checkpoint.enabled,
            enable_progress_bar=True,
            enable_model_summary=True,
            gradient_clip_val=self.config.trainer.get('gradient_clip_val', None),
            gradient_clip_algorithm=self.config.trainer.get('gradient_clip_algorithm', 'norm'),
            accumulate_grad_batches=self.config.trainer.get('accumulate_grad_batches', 1),

            # Reproducibility
            deterministic=self.config.reproducibility.deterministic,
            benchmark=self.config.reproducibility.benchmark,

            # Debugging
            fast_dev_run=self.config.trainer.get('fast_dev_run', False),
            overfit_batches=self.config.trainer.get('overfit_batches', 0.0),
            track_grad_norm=self.config.trainer.get('track_grad_norm', -1),
            detect_anomaly=self.config.framework.debug,

            # Profiling
            profiler=self.config.trainer.get('profiler', None),

            # Paths
            default_root_dir=self.config.checkpoint.dir
        )

        logger.info(f"Trainer initialized with {accelerator} accelerator and {devices} devices")

    def _get_accelerator(self) -> str:
        """
        Determine the accelerator to use.

        Returns:
            Accelerator string
        """
        if self.config.distributed.accelerator != 'auto':
            return self.config.distributed.accelerator

        if torch.cuda.is_available():
            return 'gpu'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def _get_devices(self) -> Union[int, List[int], str]:
        """
        Determine the devices to use.

        Returns:
            Device specification
        """
        if self.config.distributed.devices != 'auto':
            return self.config.distributed.devices

        accelerator = self._get_accelerator()
        if accelerator == 'gpu':
            return torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            return 1

    def _get_strategy(self) -> Optional[str]:
        """
        Determine the training strategy.

        Returns:
            Strategy string or None
        """
        if not self.config.distributed.enabled:
            return None

        strategy = self.config.distributed.strategy
        if strategy == 'auto':
            devices = self._get_devices()
            if isinstance(devices, int) and devices > 1:
                return 'ddp'
            else:
                return None

        return strategy

    def load_model(self, model_name: Optional[str] = None) -> pl.LightningModule:
        """
        Load a model from the registry.

        Args:
            model_name: Name of the model to load

        Returns:
            Model instance
        """
        model_name = model_name or self.config.model.name

        # Get model class from registry
        model_class = registry.get_model(model_name)

        # Create model instance with config
        model_config = OmegaConf.to_container(self.config.model, resolve=True)
        self.model = model_class(**model_config)

        logger.info(f"Loaded model: {model_name}")
        return self.model

    def load_datamodule(self, dataset_name: Optional[str] = None) -> pl.LightningDataModule:
        """
        Load a data module from the registry.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            DataModule instance
        """
        dataset_name = dataset_name or self.config.dataset.name

        # Get dataset class from registry
        dataset_class = registry.get_dataset(dataset_name)

        # Create dataset instance with config
        dataset_config = OmegaConf.to_container(self.config.dataset, resolve=True)
        self.datamodule = dataset_class(**dataset_config)

        logger.info(f"Loaded dataset: {dataset_name}")
        return self.datamodule

    def train(
        self,
        model: Optional[pl.LightningModule] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        ckpt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            model: Optional model to train
            datamodule: Optional data module
            ckpt_path: Optional checkpoint path to resume from

        Returns:
            Training results
        """
        # Use provided or loaded model/datamodule
        model = model or self.model
        datamodule = datamodule or self.datamodule

        if not model:
            model = self.load_model()
        if not datamodule:
            datamodule = self.load_datamodule()

        # Start MLflow run
        if self.mlflow_tracker:
            run_id = self.mlflow_tracker.start_run(
                run_name=self.config.mlops.tracking.run_name,
                tags=dict(self.config.experiment.tags) if self.config.experiment.tags else None
            )

            # Log hyperparameters
            self.mlflow_tracker.log_params(OmegaConf.to_container(self.config, resolve=True))

        try:
            # Train the model
            logger.info("Starting training...")
            self.trainer.fit(model, datamodule, ckpt_path=ckpt_path)

            # Test if enabled
            if self.config.testing.enabled:
                logger.info("Starting testing...")
                test_results = self.trainer.test(
                    model,
                    datamodule,
                    ckpt_path='best' if self.config.testing.best_checkpoint else None
                )
            else:
                test_results = None

            # Get results
            results = {
                'train_metrics': self.trainer.callback_metrics,
                'test_metrics': test_results[0] if test_results else None,
                'best_model_path': self.trainer.checkpoint_callback.best_model_path if hasattr(self.trainer, 'checkpoint_callback') else None
            }

            # Log final metrics
            if self.mlflow_tracker:
                final_metrics = {k: float(v) for k, v in self.trainer.callback_metrics.items()}
                self.mlflow_tracker.log_metrics(final_metrics)

            logger.info("Training completed successfully")
            return results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # End MLflow run
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run()

    def validate(
        self,
        model: Optional[pl.LightningModule] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        ckpt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate the model.

        Args:
            model: Optional model to validate
            datamodule: Optional data module
            ckpt_path: Optional checkpoint path

        Returns:
            Validation results
        """
        model = model or self.model
        datamodule = datamodule or self.datamodule

        if not model:
            model = self.load_model()
        if not datamodule:
            datamodule = self.load_datamodule()

        logger.info("Starting validation...")
        val_results = self.trainer.validate(model, datamodule, ckpt_path=ckpt_path)

        return val_results[0] if val_results else None

    def test(
        self,
        model: Optional[pl.LightningModule] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        ckpt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test the model.

        Args:
            model: Optional model to test
            datamodule: Optional data module
            ckpt_path: Optional checkpoint path

        Returns:
            Test results
        """
        model = model or self.model
        datamodule = datamodule or self.datamodule

        if not model:
            model = self.load_model()
        if not datamodule:
            datamodule = self.load_datamodule()

        logger.info("Starting testing...")
        test_results = self.trainer.test(model, datamodule, ckpt_path=ckpt_path)

        return test_results[0] if test_results else None

    def predict(
        self,
        model: Optional[pl.LightningModule] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        ckpt_path: Optional[str] = None
    ) -> List[Any]:
        """
        Run prediction with the model.

        Args:
            model: Optional model to use
            datamodule: Optional data module
            ckpt_path: Optional checkpoint path

        Returns:
            Predictions
        """
        model = model or self.model
        datamodule = datamodule or self.datamodule

        if not model:
            model = self.load_model()
        if not datamodule:
            datamodule = self.load_datamodule()

        logger.info("Starting prediction...")
        predictions = self.trainer.predict(model, datamodule, ckpt_path=ckpt_path)

        return predictions