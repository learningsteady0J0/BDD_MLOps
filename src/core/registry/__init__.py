"""Plugin registration system for dynamic model and dataset discovery."""

from .model_registry import ModelRegistry, register_model
from .data_registry import DataModuleRegistry, register_datamodule
from .transform_registry import TransformRegistry, register_transform
from .callback_registry import CallbackRegistry, register_callback

__all__ = [
    "ModelRegistry",
    "DataModuleRegistry",
    "TransformRegistry",
    "CallbackRegistry",
    "register_model",
    "register_datamodule",
    "register_transform",
    "register_callback",
]