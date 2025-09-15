"""Configuration manager for GUI application."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manager for loading and saving configuration files."""

    def __init__(self):
        """Initialize the configuration manager."""
        self.config_dir = Path("configs/gui")
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self, filepath: str) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            filepath: Path to configuration file

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        # Determine file format
        suffix = path.suffix.lower()

        if suffix == ".json":
            with open(path, 'r') as f:
                return json.load(f)
        elif suffix in [".yaml", ".yml"]:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {suffix}")

    def save_config(self, config: Dict[str, Any], filepath: str):
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary
            filepath: Path to save configuration

        Raises:
            ValueError: If file format is not supported
        """
        path = Path(filepath)

        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine file format
        suffix = path.suffix.lower()

        if suffix == ".json":
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        elif suffix in [".yaml", ".yml"]:
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration format: {suffix}")

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "experiment": {
                "name": "default_experiment",
                "seed": 42,
                "description": ""
            },
            "model": {
                "name": "resnet",
                "variant": "resnet50",
                "num_classes": 10,
                "pretrained": True,
                "freeze_backbone": False,
                "dropout_rate": 0.5
            },
            "data": {
                "name": "cifar10",
                "data_dir": "./data",
                "batch_size": 32,
                "num_workers": 4,
                "val_split": 0.2,
                "pin_memory": True,
                "persistent_workers": True,
                "shuffle_train": True,
                "drop_last": False,
                "augmentation": {
                    "preset": "standard",
                    "random_crop": True,
                    "random_flip": True,
                    "color_jitter": True,
                    "random_rotation": False,
                    "normalize": True,
                    "mixup": False,
                    "cutmix": False,
                    "rotation_degrees": 15,
                    "jitter_brightness": 0.2,
                    "mixup_alpha": 0.2
                }
            },
            "training": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "momentum": 0.9,
                "max_epochs": 100,
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": 1,
                "precision": 16,
                "scheduler": {
                    "type": "cosine",
                    "step_size": 10,
                    "gamma": 0.1
                },
                "early_stopping": {
                    "monitor": "val_loss",
                    "patience": 10,
                    "min_delta": 0.0001,
                    "mode": "min"
                },
                "checkpoint": {
                    "save_top_k": 3,
                    "save_every_n_epochs": 1,
                    "save_last": True
                }
            },
            "hardware": {
                "accelerator": "auto",
                "devices": "auto",
                "precision": 16,
                "num_workers": 4
            },
            "paths": {
                "output_dir": "./outputs",
                "log_dir": "./logs",
                "checkpoint_dir": "./checkpoints"
            }
        }

    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        required_fields = [
            ("model", "name"),
            ("data", "name"),
            ("training", "max_epochs"),
            ("training", "learning_rate")
        ]

        for section, field in required_fields:
            if section not in config:
                errors.append(f"Missing section: {section}")
            elif field not in config[section]:
                errors.append(f"Missing field: {section}.{field}")

        # Validate value ranges
        if "training" in config:
            training = config["training"]

            if "max_epochs" in training:
                if training["max_epochs"] <= 0:
                    errors.append("max_epochs must be greater than 0")

            if "learning_rate" in training:
                if training["learning_rate"] <= 0:
                    errors.append("learning_rate must be greater than 0")

            if "batch_size" in config.get("data", {}):
                if config["data"]["batch_size"] <= 0:
                    errors.append("batch_size must be greater than 0")

        return len(errors) == 0, errors

    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configurations, with override taking precedence.

        Args:
            base: Base configuration
            override: Configuration to override with

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def get_recent_configs(self, max_items: int = 10) -> list[Path]:
        """
        Get list of recent configuration files.

        Args:
            max_items: Maximum number of items to return

        Returns:
            List of configuration file paths
        """
        config_files = []

        # Search for config files
        for pattern in ["*.json", "*.yaml", "*.yml"]:
            config_files.extend(self.config_dir.glob(pattern))

        # Sort by modification time
        config_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return config_files[:max_items]

    def create_preset(self, name: str, config: Dict[str, Any]):
        """
        Create a configuration preset.

        Args:
            name: Preset name
            config: Configuration to save as preset
        """
        preset_dir = self.config_dir / "presets"
        preset_dir.mkdir(parents=True, exist_ok=True)

        preset_file = preset_dir / f"{name}.yaml"
        self.save_config(config, str(preset_file))

    def load_preset(self, name: str) -> Dict[str, Any]:
        """
        Load a configuration preset.

        Args:
            name: Preset name

        Returns:
            Preset configuration

        Raises:
            FileNotFoundError: If preset doesn't exist
        """
        preset_file = self.config_dir / "presets" / f"{name}.yaml"

        if not preset_file.exists():
            raise FileNotFoundError(f"Preset not found: {name}")

        return self.load_config(str(preset_file))

    def list_presets(self) -> list[str]:
        """
        List available presets.

        Returns:
            List of preset names
        """
        preset_dir = self.config_dir / "presets"

        if not preset_dir.exists():
            return []

        presets = []
        for file in preset_dir.glob("*.yaml"):
            presets.append(file.stem)

        return sorted(presets)