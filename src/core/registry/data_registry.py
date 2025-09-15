"""DataModule registration system for dynamic dataset discovery."""

from typing import Type, Dict, Any, Optional, Callable, List
from functools import wraps
import inspect
from pathlib import Path

from omegaconf import DictConfig
import pytorch_lightning as pl


class DataModuleRegistry:
    """
    Registry for vision data modules with automatic discovery.

    Features:
    - Dynamic datamodule registration
    - Dataset metadata tracking
    - Factory pattern for dataset instantiation
    - Dataset compatibility checking
    """

    _datamodules: Dict[str, Dict[str, Any]] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(
        cls,
        name: str,
        dataset_type: str = "image_classification",
        input_size: Optional[tuple] = None,
        num_classes: Optional[int] = None,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Register a datamodule class.

        Args:
            name: Unique name for the datamodule
            dataset_type: Type of dataset
            input_size: Expected input size
            num_classes: Number of classes
            aliases: Alternative names
            metadata: Additional metadata

        Returns:
            Decorator function
        """
        def decorator(datamodule_class: Type[pl.LightningDataModule]) -> Type[pl.LightningDataModule]:
            if not issubclass(datamodule_class, pl.LightningDataModule):
                raise TypeError(f"{datamodule_class.__name__} must inherit from pl.LightningDataModule")

            cls._datamodules[name] = {
                "class": datamodule_class,
                "dataset_type": dataset_type,
                "input_size": input_size,
                "num_classes": num_classes,
                "metadata": metadata or {},
                "module": datamodule_class.__module__,
                "docstring": inspect.getdoc(datamodule_class),
                "signature": inspect.signature(datamodule_class.__init__)
            }

            if aliases:
                for alias in aliases:
                    cls._aliases[alias] = name

            return datamodule_class

        return decorator

    @classmethod
    def get_datamodule_class(cls, name: str) -> Type[pl.LightningDataModule]:
        """Get a datamodule class by name."""
        if name in cls._aliases:
            name = cls._aliases[name]

        if name not in cls._datamodules:
            available = ", ".join(cls._datamodules.keys())
            raise KeyError(f"DataModule '{name}' not found. Available: {available}")

        return cls._datamodules[name]["class"]

    @classmethod
    def create_datamodule(
        cls,
        name: str,
        config: DictConfig,
        **kwargs
    ) -> pl.LightningDataModule:
        """Create a datamodule instance from configuration."""
        datamodule_class = cls.get_datamodule_class(name)
        return datamodule_class(config=config, **kwargs)

    @classmethod
    def list_datamodules(
        cls,
        dataset_type: Optional[str] = None,
        verbose: bool = False
    ) -> List[Any]:
        """List all registered datamodules."""
        datamodules = cls._datamodules

        if dataset_type:
            datamodules = {
                name: info
                for name, info in datamodules.items()
                if info["dataset_type"] == dataset_type
            }

        if verbose:
            return [
                {
                    "name": name,
                    "dataset_type": info["dataset_type"],
                    "input_size": info["input_size"],
                    "num_classes": info["num_classes"],
                    "module": info["module"]
                }
                for name, info in datamodules.items()
            ]
        else:
            return list(datamodules.keys())

    @classmethod
    def get_datamodule_info(cls, name: str) -> Dict[str, Any]:
        """Get detailed information about a datamodule."""
        if name in cls._aliases:
            name = cls._aliases[name]

        if name not in cls._datamodules:
            raise KeyError(f"DataModule '{name}' not found")

        info = cls._datamodules[name].copy()
        info.pop("class")
        return info

    @classmethod
    def check_compatibility(cls, model_name: str, datamodule_name: str) -> bool:
        """
        Check if a model and datamodule are compatible.

        Args:
            model_name: Name of the model
            datamodule_name: Name of the datamodule

        Returns:
            True if compatible
        """
        from .model_registry import ModelRegistry

        model_info = ModelRegistry.get_model_info(model_name)
        datamodule_info = cls.get_datamodule_info(datamodule_name)

        # Check task type compatibility
        model_task = model_info.get("task_type")
        data_type = datamodule_info.get("dataset_type")

        compatibility_map = {
            "classification": ["image_classification", "multi_label_classification"],
            "detection": ["object_detection", "instance_detection"],
            "segmentation": ["semantic_segmentation", "instance_segmentation"]
        }

        if model_task in compatibility_map:
            return data_type in compatibility_map[model_task]

        return True  # Default to compatible if unsure

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered datamodules."""
        cls._datamodules.clear()
        cls._aliases.clear()


def register_datamodule(
    name: str,
    dataset_type: str = "image_classification",
    input_size: Optional[tuple] = None,
    num_classes: Optional[int] = None,
    aliases: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Decorator to register a datamodule.

    Usage:
        @register_datamodule("cifar10", dataset_type="image_classification", num_classes=10)
        class CIFAR10DataModule(BaseVisionDataModule):
            ...
    """
    return DataModuleRegistry.register(
        name, dataset_type, input_size, num_classes, aliases, metadata
    )