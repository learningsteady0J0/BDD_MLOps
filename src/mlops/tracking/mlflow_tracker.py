"""MLflow integration for experiment tracking with MongoDB backend."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import logging
from datetime import datetime
from contextlib import contextmanager

from src.core.contracts.interfaces import IExperimentTracker

logger = logging.getLogger(__name__)


class MLflowTracker(IExperimentTracker):
    """
    MLflow experiment tracker with MongoDB backend support.

    This class provides a comprehensive interface for experiment tracking,
    model versioning, and artifact management using MLflow with MongoDB
    as the backend store.
    """

    def __init__(
        self,
        tracking_uri: str = "mongodb://localhost:27017",
        default_experiment: str = "Default",
        artifact_location: Optional[str] = None,
        registry_uri: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking server URI (MongoDB connection string)
            default_experiment: Default experiment name
            artifact_location: Location for storing artifacts
            registry_uri: Model registry URI
        """
        self.tracking_uri = tracking_uri
        self.default_experiment = default_experiment
        self.artifact_location = artifact_location or "./mlruns"
        self.registry_uri = registry_uri or tracking_uri

        # Set MLflow configurations
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)

        # Initialize MLflow client
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        # Create default experiment if it doesn't exist
        self._ensure_experiment_exists(self.default_experiment)

        # Set default experiment
        mlflow.set_experiment(self.default_experiment)

        # Current run tracking
        self.current_run = None
        self.run_stack = []

    def _ensure_experiment_exists(self, experiment_name: str) -> None:
        """
        Ensure an experiment exists, create if not.

        Args:
            experiment_name: Name of the experiment
        """
        try:
            self.client.get_experiment_by_name(experiment_name)
        except Exception:
            self.client.create_experiment(
                name=experiment_name,
                artifact_location=f"{self.artifact_location}/{experiment_name}"
            )
            logger.info(f"Created experiment: {experiment_name}")

    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
        experiment_name: Optional[str] = None,
        nested: bool = False
    ) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Optional tags for the run
            experiment_name: Optional experiment name
            nested: Whether this is a nested run

        Returns:
            Run ID
        """
        if experiment_name:
            self._ensure_experiment_exists(experiment_name)
            mlflow.set_experiment(experiment_name)

        # Add default tags
        if tags is None:
            tags = {}
        tags.update({
            "framework": "pytorch_lightning",
            "timestamp": datetime.now().isoformat(),
            "user": os.getenv("USER", "unknown")
        })

        # Start the run
        self.current_run = mlflow.start_run(
            run_name=run_name,
            tags=tags,
            nested=nested
        )

        # Track run in stack for nested runs
        if nested:
            self.run_stack.append(self.current_run)

        logger.info(f"Started MLflow run: {self.current_run.info.run_id}")
        return self.current_run.info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.

        Args:
            params: Parameters to log
        """
        if not self.current_run:
            raise RuntimeError("No active run. Call start_run() first.")

        # Flatten nested parameters
        flat_params = self._flatten_dict(params)

        # Log parameters (MLflow has a limit on param value length)
        for key, value in flat_params.items():
            try:
                # Convert value to string and truncate if necessary
                str_value = str(value)[:250]  # MLflow param value limit
                mlflow.log_param(key, str_value)
            except Exception as e:
                logger.warning(f"Failed to log parameter {key}: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log metrics to the current run.

        Args:
            metrics: Metrics to log
            step: Optional step number
        """
        if not self.current_run:
            raise RuntimeError("No active run. Call start_run() first.")

        # Log each metric
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric {key}: {e}")

    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: Optional[str] = None
    ) -> None:
        """
        Log an artifact to the current run.

        Args:
            artifact_path: Path to the artifact
            artifact_type: Optional type of artifact
        """
        if not self.current_run:
            raise RuntimeError("No active run. Call start_run() first.")

        try:
            if os.path.isfile(artifact_path):
                mlflow.log_artifact(artifact_path)
            elif os.path.isdir(artifact_path):
                mlflow.log_artifacts(artifact_path)

            if artifact_type:
                mlflow.set_tag(f"artifact_{Path(artifact_path).name}_type", artifact_type)

            logger.info(f"Logged artifact: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact {artifact_path}: {e}")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        **kwargs
    ) -> None:
        """
        Log a model to the current run.

        Args:
            model: Model to log
            artifact_path: Path to save the model
            registered_model_name: Optional name for model registry
            signature: Optional model signature
            input_example: Optional input example
            **kwargs: Additional arguments for model logging
        """
        if not self.current_run:
            raise RuntimeError("No active run. Call start_run() first.")

        try:
            # Determine model flavor
            if hasattr(model, "save_pretrained"):  # Transformers
                mlflow.transformers.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    signature=signature,
                    input_example=input_example,
                    **kwargs
                )
            elif hasattr(model, "save"):  # PyTorch
                mlflow.pytorch.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    signature=signature,
                    input_example=input_example,
                    **kwargs
                )
            else:
                # Generic Python model
                mlflow.pyfunc.log_model(
                    artifact_path,
                    python_model=model,
                    registered_model_name=registered_model_name,
                    signature=signature,
                    input_example=input_example,
                    **kwargs
                )

            logger.info(f"Logged model to {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def end_run(self, status: Optional[str] = None) -> None:
        """
        End the current run.

        Args:
            status: Optional status for the run
        """
        if not self.current_run:
            logger.warning("No active run to end")
            return

        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.current_run.info.run_id}")

            # Pop from stack if nested
            if self.run_stack and self.run_stack[-1] == self.current_run:
                self.run_stack.pop()

            # Set current run to previous in stack or None
            self.current_run = self.run_stack[-1] if self.run_stack else None
        except Exception as e:
            logger.error(f"Failed to end run: {e}")

    def get_run_id(self) -> str:
        """
        Get the current run ID.

        Returns:
            Current run ID
        """
        if not self.current_run:
            raise RuntimeError("No active run")
        return self.current_run.info.run_id

    def get_experiment_id(self, experiment_name: str) -> str:
        """
        Get experiment ID by name.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Experiment ID
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        return experiment.experiment_id if experiment else None

    def search_runs(
        self,
        experiment_names: Optional[List[str]] = None,
        filter_string: Optional[str] = None,
        max_results: int = 100,
        order_by: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for runs.

        Args:
            experiment_names: List of experiment names to search
            filter_string: Filter string for search
            max_results: Maximum number of results
            order_by: List of columns to order by

        Returns:
            List of run dictionaries
        """
        experiment_ids = []
        if experiment_names:
            for name in experiment_names:
                exp_id = self.get_experiment_id(name)
                if exp_id:
                    experiment_ids.append(exp_id)
        else:
            # Search all experiments
            experiments = self.client.search_experiments()
            experiment_ids = [exp.experiment_id for exp in experiments]

        runs = self.client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=max_results,
            order_by=order_by or ["metrics.val_loss ASC"]
        )

        return [self._run_to_dict(run) for run in runs]

    def get_best_run(
        self,
        experiment_name: str,
        metric: str,
        mode: str = "min"
    ) -> Dict[str, Any]:
        """
        Get the best run based on a metric.

        Args:
            experiment_name: Name of the experiment
            metric: Metric to optimize
            mode: 'min' or 'max'

        Returns:
            Best run dictionary
        """
        order = f"metrics.{metric} {'ASC' if mode == 'min' else 'DESC'}"
        runs = self.search_runs(
            experiment_names=[experiment_name],
            max_results=1,
            order_by=[order]
        )
        return runs[0] if runs else None

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "."
    ) -> Dict[str, Any]:
        """
        Flatten a nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items
            sep: Separator for keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _run_to_dict(self, run) -> Dict[str, Any]:
        """
        Convert MLflow run to dictionary.

        Args:
            run: MLflow run object

        Returns:
            Dictionary representation
        """
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags
        }

    @contextmanager
    def track_run(
        self,
        run_name: str,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Context manager for tracking a run.

        Args:
            run_name: Name for the run
            experiment_name: Optional experiment name
            tags: Optional tags

        Yields:
            Run ID
        """
        run_id = self.start_run(run_name, tags, experiment_name)
        try:
            yield run_id
            self.end_run(status="FINISHED")
        except Exception as e:
            self.end_run(status="FAILED")
            raise e

    def delete_run(self, run_id: str) -> None:
        """
        Delete a run.

        Args:
            run_id: ID of the run to delete
        """
        self.client.delete_run(run_id)
        logger.info(f"Deleted run: {run_id}")

    def restore_run(self, run_id: str) -> None:
        """
        Restore a deleted run.

        Args:
            run_id: ID of the run to restore
        """
        self.client.restore_run(run_id)
        logger.info(f"Restored run: {run_id}")