"""Transform registration system for augmentation pipeline."""

from typing import Type, Dict, Any, Optional, Callable, List
from functools import wraps
import inspect

from omegaconf import DictConfig


class TransformRegistry:
    """Registry for vision transforms and augmentations."""

    _transforms: Dict[str, Dict[str, Any]] = {}
    _pipelines: Dict[str, List[str]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        transform_type: str = "augmentation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """Register a transform class."""
        def decorator(transform_class: Type) -> Type:
            cls._transforms[name] = {
                "class": transform_class,
                "transform_type": transform_type,
                "metadata": metadata or {},
                "module": transform_class.__module__,
                "docstring": inspect.getdoc(transform_class),
                "signature": inspect.signature(transform_class.__init__)
            }
            return transform_class
        return decorator

    @classmethod
    def get_transform_class(cls, name: str) -> Type:
        """Get a transform class by name."""
        if name not in cls._transforms:
            available = ", ".join(cls._transforms.keys())
            raise KeyError(f"Transform '{name}' not found. Available: {available}")
        return cls._transforms[name]["class"]

    @classmethod
    def create_transform(cls, name: str, **kwargs) -> Any:
        """Create a transform instance."""
        transform_class = cls.get_transform_class(name)
        return transform_class(**kwargs)

    @classmethod
    def register_pipeline(cls, name: str, transforms: List[str]) -> None:
        """Register a transform pipeline."""
        cls._pipelines[name] = transforms

    @classmethod
    def get_pipeline(cls, name: str) -> List[Any]:
        """Get a transform pipeline by name."""
        if name not in cls._pipelines:
            raise KeyError(f"Pipeline '{name}' not found")

        transforms = []
        for transform_name in cls._pipelines[name]:
            transforms.append(cls.get_transform_class(transform_name))
        return transforms

    @classmethod
    def list_transforms(cls, transform_type: Optional[str] = None) -> List[str]:
        """List all registered transforms."""
        if transform_type:
            return [
                name for name, info in cls._transforms.items()
                if info["transform_type"] == transform_type
            ]
        return list(cls._transforms.keys())

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered transforms."""
        cls._transforms.clear()
        cls._pipelines.clear()


def register_transform(
    name: str,
    transform_type: str = "augmentation",
    metadata: Optional[Dict[str, Any]] = None
) -> Callable:
    """Decorator to register a transform."""
    return TransformRegistry.register(name, transform_type, metadata)