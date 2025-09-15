"""Plugin registry system for dynamic component discovery and registration."""

import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for registered plugins."""
    name: str
    version: str
    description: str
    author: str
    domain: str
    entry_point: str
    dependencies: List[str]


class PluginRegistry:
    """
    Central registry for all plugins in the framework.

    This registry manages discovery, registration, and retrieval of plugins
    across different domains (models, datasets, transforms, etc.).
    """

    def __init__(self):
        """Initialize the plugin registry."""
        self._models: Dict[str, Type] = {}
        self._datasets: Dict[str, Type] = {}
        self._transforms: Dict[str, Type] = {}
        self._callbacks: Dict[str, Type] = {}
        self._metrics: Dict[str, Type] = {}
        self._optimizers: Dict[str, Type] = {}
        self._schedulers: Dict[str, Type] = {}
        self._metadata: Dict[str, PluginMetadata] = {}

    def register_model(
        self,
        name: str,
        model_class: Type,
        metadata: Optional[PluginMetadata] = None,
        override: bool = False
    ) -> None:
        """
        Register a model plugin.

        Args:
            name: Unique name for the model
            model_class: Model class to register
            metadata: Optional metadata for the plugin
            override: Whether to override existing registration

        Raises:
            ValueError: If name already exists and override is False
        """
        if name in self._models and not override:
            raise ValueError(f"Model '{name}' already registered")

        self._models[name] = model_class
        if metadata:
            self._metadata[f"model:{name}"] = metadata

        logger.info(f"Registered model: {name}")

    def register_dataset(
        self,
        name: str,
        dataset_class: Type,
        metadata: Optional[PluginMetadata] = None,
        override: bool = False
    ) -> None:
        """
        Register a dataset plugin.

        Args:
            name: Unique name for the dataset
            dataset_class: Dataset class to register
            metadata: Optional metadata for the plugin
            override: Whether to override existing registration

        Raises:
            ValueError: If name already exists and override is False
        """
        if name in self._datasets and not override:
            raise ValueError(f"Dataset '{name}' already registered")

        self._datasets[name] = dataset_class
        if metadata:
            self._metadata[f"dataset:{name}"] = metadata

        logger.info(f"Registered dataset: {name}")

    def register_transform(
        self,
        name: str,
        transform_class: Type,
        metadata: Optional[PluginMetadata] = None,
        override: bool = False
    ) -> None:
        """
        Register a transform plugin.

        Args:
            name: Unique name for the transform
            transform_class: Transform class to register
            metadata: Optional metadata for the plugin
            override: Whether to override existing registration

        Raises:
            ValueError: If name already exists and override is False
        """
        if name in self._transforms and not override:
            raise ValueError(f"Transform '{name}' already registered")

        self._transforms[name] = transform_class
        if metadata:
            self._metadata[f"transform:{name}"] = metadata

        logger.info(f"Registered transform: {name}")

    def register_callback(
        self,
        name: str,
        callback_class: Type,
        metadata: Optional[PluginMetadata] = None,
        override: bool = False
    ) -> None:
        """
        Register a callback plugin.

        Args:
            name: Unique name for the callback
            callback_class: Callback class to register
            metadata: Optional metadata for the plugin
            override: Whether to override existing registration
        """
        if name in self._callbacks and not override:
            raise ValueError(f"Callback '{name}' already registered")

        self._callbacks[name] = callback_class
        if metadata:
            self._metadata[f"callback:{name}"] = metadata

        logger.info(f"Registered callback: {name}")

    def get_model(self, name: str) -> Type:
        """
        Get a registered model class.

        Args:
            name: Name of the model

        Returns:
            Model class

        Raises:
            KeyError: If model not found
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return self._models[name]

    def get_dataset(self, name: str) -> Type:
        """
        Get a registered dataset class.

        Args:
            name: Name of the dataset

        Returns:
            Dataset class

        Raises:
            KeyError: If dataset not found
        """
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found. Available: {list(self._datasets.keys())}")
        return self._datasets[name]

    def get_transform(self, name: str) -> Type:
        """
        Get a registered transform class.

        Args:
            name: Name of the transform

        Returns:
            Transform class

        Raises:
            KeyError: If transform not found
        """
        if name not in self._transforms:
            raise KeyError(f"Transform '{name}' not found. Available: {list(self._transforms.keys())}")
        return self._transforms[name]

    def get_callback(self, name: str) -> Type:
        """
        Get a registered callback class.

        Args:
            name: Name of the callback

        Returns:
            Callback class

        Raises:
            KeyError: If callback not found
        """
        if name not in self._callbacks:
            raise KeyError(f"Callback '{name}' not found. Available: {list(self._callbacks.keys())}")
        return self._callbacks[name]

    def list_models(self, domain: Optional[str] = None) -> List[str]:
        """
        List all registered models.

        Args:
            domain: Optional domain filter

        Returns:
            List of model names
        """
        if domain:
            return [
                name for name in self._models.keys()
                if self._get_domain(f"model:{name}") == domain
            ]
        return list(self._models.keys())

    def list_datasets(self, domain: Optional[str] = None) -> List[str]:
        """
        List all registered datasets.

        Args:
            domain: Optional domain filter

        Returns:
            List of dataset names
        """
        if domain:
            return [
                name for name in self._datasets.keys()
                if self._get_domain(f"dataset:{name}") == domain
            ]
        return list(self._datasets.keys())

    def _get_domain(self, key: str) -> Optional[str]:
        """Get domain for a plugin."""
        if key in self._metadata:
            return self._metadata[key].domain
        return None

    def discover_plugins(self, plugin_dir: Union[str, Path]) -> None:
        """
        Discover and register plugins from a directory.

        Args:
            plugin_dir: Directory containing plugins
        """
        plugin_dir = Path(plugin_dir)
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {plugin_dir}")
            return

        for plugin_file in plugin_dir.glob("**/*.py"):
            if plugin_file.name.startswith("_"):
                continue

            try:
                self._load_plugin_from_file(plugin_file)
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")

    def _load_plugin_from_file(self, file_path: Path) -> None:
        """
        Load a plugin from a Python file.

        Args:
            file_path: Path to the plugin file
        """
        # Convert file path to module path
        module_path = str(file_path.with_suffix("")).replace("/", ".").replace("\\", ".")

        # Import the module
        spec = importlib.util.spec_from_file_location(module_path, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find and register plugins in the module
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and hasattr(obj, "__plugin_type__"):
                    self._register_plugin_class(name, obj)

    def _register_plugin_class(self, name: str, plugin_class: Type) -> None:
        """
        Register a plugin class based on its type.

        Args:
            name: Name of the plugin
            plugin_class: Plugin class to register
        """
        plugin_type = getattr(plugin_class, "__plugin_type__", None)
        plugin_name = getattr(plugin_class, "__plugin_name__", name)

        if plugin_type == "model":
            self.register_model(plugin_name, plugin_class)
        elif plugin_type == "dataset":
            self.register_dataset(plugin_name, plugin_class)
        elif plugin_type == "transform":
            self.register_transform(plugin_name, plugin_class)
        elif plugin_type == "callback":
            self.register_callback(plugin_name, plugin_class)

    def create_model(self, name: str, **kwargs) -> Any:
        """
        Create an instance of a registered model.

        Args:
            name: Name of the model
            **kwargs: Arguments to pass to the model constructor

        Returns:
            Model instance
        """
        model_class = self.get_model(name)
        return model_class(**kwargs)

    def create_dataset(self, name: str, **kwargs) -> Any:
        """
        Create an instance of a registered dataset.

        Args:
            name: Name of the dataset
            **kwargs: Arguments to pass to the dataset constructor

        Returns:
            Dataset instance
        """
        dataset_class = self.get_dataset(name)
        return dataset_class(**kwargs)

    def clear(self) -> None:
        """Clear all registered plugins."""
        self._models.clear()
        self._datasets.clear()
        self._transforms.clear()
        self._callbacks.clear()
        self._metrics.clear()
        self._optimizers.clear()
        self._schedulers.clear()
        self._metadata.clear()
        logger.info("Cleared all registered plugins")


# Global registry instance
registry = PluginRegistry()


def register_model(name: Optional[str] = None, **metadata):
    """
    Decorator to register a model class.

    Args:
        name: Optional name for the model
        **metadata: Additional metadata

    Returns:
        Decorator function
    """
    def decorator(cls):
        model_name = name or cls.__name__
        cls.__plugin_type__ = "model"
        cls.__plugin_name__ = model_name
        registry.register_model(model_name, cls)
        return cls
    return decorator


def register_dataset(name: Optional[str] = None, **metadata):
    """
    Decorator to register a dataset class.

    Args:
        name: Optional name for the dataset
        **metadata: Additional metadata

    Returns:
        Decorator function
    """
    def decorator(cls):
        dataset_name = name or cls.__name__
        cls.__plugin_type__ = "dataset"
        cls.__plugin_name__ = dataset_name
        registry.register_dataset(dataset_name, cls)
        return cls
    return decorator


def register_transform(name: Optional[str] = None, **metadata):
    """
    Decorator to register a transform class.

    Args:
        name: Optional name for the transform
        **metadata: Additional metadata

    Returns:
        Decorator function
    """
    def decorator(cls):
        transform_name = name or cls.__name__
        cls.__plugin_type__ = "transform"
        cls.__plugin_name__ = transform_name
        registry.register_transform(transform_name, cls)
        return cls
    return decorator


def register_callback(name: Optional[str] = None, **metadata):
    """
    Decorator to register a callback class.

    Args:
        name: Optional name for the callback
        **metadata: Additional metadata

    Returns:
        Decorator function
    """
    def decorator(cls):
        callback_name = name or cls.__name__
        cls.__plugin_type__ = "callback"
        cls.__plugin_name__ = callback_name
        registry.register_callback(callback_name, cls)
        return cls
    return decorator