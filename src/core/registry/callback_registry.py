"""Callback registration system for training hooks."""

from typing import Type, Dict, Any, Optional, Callable, List
from functools import wraps
import inspect

from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig


class CallbackRegistry:
    """Registry for training callbacks."""

    _callbacks: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        callback_type: str = "training",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """Register a callback class."""
        def decorator(callback_class: Type[Callback]) -> Type[Callback]:
            if not issubclass(callback_class, Callback):
                raise TypeError(f"{callback_class.__name__} must inherit from Callback")

            cls._callbacks[name] = {
                "class": callback_class,
                "callback_type": callback_type,
                "metadata": metadata or {},
                "module": callback_class.__module__,
                "docstring": inspect.getdoc(callback_class),
                "signature": inspect.signature(callback_class.__init__)
            }
            return callback_class
        return decorator

    @classmethod
    def get_callback_class(cls, name: str) -> Type[Callback]:
        """Get a callback class by name."""
        if name not in cls._callbacks:
            available = ", ".join(cls._callbacks.keys())
            raise KeyError(f"Callback '{name}' not found. Available: {available}")
        return cls._callbacks[name]["class"]

    @classmethod
    def create_callback(cls, name: str, config: Optional[DictConfig] = None, **kwargs) -> Callback:
        """Create a callback instance."""
        callback_class = cls.get_callback_class(name)
        if config:
            return callback_class(config=config, **kwargs)
        return callback_class(**kwargs)

    @classmethod
    def list_callbacks(cls, callback_type: Optional[str] = None) -> List[str]:
        """List all registered callbacks."""
        if callback_type:
            return [
                name for name, info in cls._callbacks.items()
                if info["callback_type"] == callback_type
            ]
        return list(cls._callbacks.keys())

    @classmethod
    def create_callback_list(cls, callback_names: List[str], config: DictConfig) -> List[Callback]:
        """Create a list of callbacks from names."""
        callbacks = []
        for name in callback_names:
            callback_config = config.get(name, {})
            callbacks.append(cls.create_callback(name, callback_config))
        return callbacks

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered callbacks."""
        cls._callbacks.clear()


def register_callback(
    name: str,
    callback_type: str = "training",
    metadata: Optional[Dict[str, Any]] = None
) -> Callable:
    """Decorator to register a callback."""
    return CallbackRegistry.register(name, callback_type, metadata)