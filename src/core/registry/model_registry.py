"""Model registration system for dynamic model discovery and instantiation."""

from typing import Type, Dict, Any, Optional, Callable, List
from functools import wraps
import inspect
from pathlib import Path

from omegaconf import DictConfig
import pytorch_lightning as pl


class ModelRegistry:
    """
    Registry for vision models with automatic discovery and factory pattern.

    This registry allows:
    - Dynamic model registration via decorators
    - Automatic model discovery from modules
    - Factory pattern for model instantiation
    - Model versioning and metadata tracking
    """

    _models: Dict[str, Dict[str, Any]] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(
        cls,
        name: str,
        version: str = "1.0.0",
        task_type: str = "classification",
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Register a model class with the registry.

        Args:
            name: Unique name for the model
            version: Model version
            task_type: Type of vision task
            aliases: Alternative names for the model
            metadata: Additional metadata about the model

        Returns:
            Decorator function
        """
        def decorator(model_class: Type[pl.LightningModule]) -> Type[pl.LightningModule]:
            # Validate that the class is a LightningModule
            if not issubclass(model_class, pl.LightningModule):
                raise TypeError(f"{model_class.__name__} must inherit from pl.LightningModule")

            # Store model information
            cls._models[name] = {
                "class": model_class,
                "version": version,
                "task_type": task_type,
                "metadata": metadata or {},
                "module": model_class.__module__,
                "docstring": inspect.getdoc(model_class),
                "signature": inspect.signature(model_class.__init__)
            }

            # Register aliases
            if aliases:
                for alias in aliases:
                    cls._aliases[alias] = name

            return model_class

        return decorator

    @classmethod
    def get_model_class(cls, name: str) -> Type[pl.LightningModule]:
        """
        Get a model class by name.

        Args:
            name: Model name or alias

        Returns:
            Model class

        Raises:
            KeyError: If model not found
        """
        # Check if it's an alias
        if name in cls._aliases:
            name = cls._aliases[name]

        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise KeyError(f"Model '{name}' not found. Available models: {available}")

        return cls._models[name]["class"]

    @classmethod
    def create_model(
        cls,
        name: str,
        config: DictConfig,
        **kwargs
    ) -> pl.LightningModule:
        """
        Create a model instance from configuration.

        Args:
            name: Model name
            config: Model configuration
            **kwargs: Additional arguments to pass to model

        Returns:
            Model instance
        """
        model_class = cls.get_model_class(name)

        # Merge config and kwargs
        init_params = dict(config)
        init_params.update(kwargs)

        # Create model instance
        return model_class(config=config, **kwargs)

    @classmethod
    def list_models(
        cls,
        task_type: Optional[str] = None,
        verbose: bool = False
    ) -> List[str]:
        """
        List all registered models.

        Args:
            task_type: Filter by task type
            verbose: Include detailed information

        Returns:
            List of model names or detailed info
        """
        models = cls._models

        if task_type:
            models = {
                name: info
                for name, info in models.items()
                if info["task_type"] == task_type
            }

        if verbose:
            return [
                {
                    "name": name,
                    "version": info["version"],
                    "task_type": info["task_type"],
                    "module": info["module"],
                    "docstring": info["docstring"]
                }
                for name, info in models.items()
            ]
        else:
            return list(models.keys())

    @classmethod
    def get_model_info(cls, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.

        Args:
            name: Model name

        Returns:
            Model information dictionary
        """
        if name in cls._aliases:
            name = cls._aliases[name]

        if name not in cls._models:
            raise KeyError(f"Model '{name}' not found")

        info = cls._models[name].copy()
        info.pop("class")  # Don't return the actual class
        return info

    @classmethod
    def discover_models(cls, module_path: Path) -> None:
        """
        Automatically discover and register models from a module path.

        Args:
            module_path: Path to module containing models
        """
        import importlib
        import pkgutil

        # Convert path to module name
        module_name = str(module_path).replace("/", ".").replace("\\", ".")

        # Import the module
        module = importlib.import_module(module_name)

        # Iterate through submodules
        for _, name, ispkg in pkgutil.iter_modules(module.__path__):
            full_name = f"{module_name}.{name}"
            submodule = importlib.import_module(full_name)

            # Look for LightningModule subclasses
            for item_name in dir(submodule):
                item = getattr(submodule, item_name)
                if (
                    inspect.isclass(item)
                    and issubclass(item, pl.LightningModule)
                    and item != pl.LightningModule
                ):
                    # Auto-register if not already registered
                    model_name = item_name.lower()
                    if model_name not in cls._models:
                        cls.register(
                            name=model_name,
                            version="auto-discovered",
                            task_type="unknown"
                        )(item)

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered models (useful for testing)."""
        cls._models.clear()
        cls._aliases.clear()

    @classmethod
    def validate_config(cls, name: str, config: DictConfig) -> bool:
        """
        Validate if a configuration is compatible with a model.

        Args:
            name: Model name
            config: Configuration to validate

        Returns:
            True if valid, raises exception otherwise
        """
        model_info = cls.get_model_info(name)
        signature = model_info["signature"]

        # Check required parameters
        required_params = [
            param.name
            for param in signature.parameters.values()
            if param.default == inspect.Parameter.empty
            and param.name not in ["self", "config"]
        ]

        for param in required_params:
            if param not in config:
                raise ValueError(f"Required parameter '{param}' missing in config for model '{name}'")

        return True


def register_model(
    name: str,
    version: str = "1.0.0",
    task_type: str = "classification",
    aliases: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Decorator to register a model with the global registry.

    Usage:
        @register_model("resnet18", task_type="classification", aliases=["rn18"])
        class ResNet18(BaseVisionModel):
            ...

    Args:
        name: Unique name for the model
        version: Model version
        task_type: Type of vision task
        aliases: Alternative names
        metadata: Additional metadata

    Returns:
        Decorator function
    """
    return ModelRegistry.register(name, version, task_type, aliases, metadata)