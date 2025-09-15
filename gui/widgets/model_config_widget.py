"""Widget for model configuration."""

from typing import Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QGridLayout, QLineEdit
)
from PyQt5.QtCore import pyqtSignal, Qt


class ModelConfigWidget(QGroupBox):
    """Widget for configuring model settings."""

    # Signal emitted when configuration changes
    config_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        """Initialize the model configuration widget."""
        super().__init__("Model Configuration", parent)
        self.init_ui()
        self.setup_connections()
        self.set_default_values()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Model selection
        model_layout = QGridLayout()

        # Model type
        model_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["ResNet", "VGG", "EfficientNet"])
        model_layout.addWidget(self.model_type_combo, 0, 1)

        # Model variant
        model_layout.addWidget(QLabel("Model Variant:"), 1, 0)
        self.model_variant_combo = QComboBox()
        self.update_model_variants()
        model_layout.addWidget(self.model_variant_combo, 1, 1)

        # Number of classes
        model_layout.addWidget(QLabel("Number of Classes:"), 2, 0)
        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(2, 1000)
        self.num_classes_spin.setValue(10)
        model_layout.addWidget(self.num_classes_spin, 2, 1)

        # Pretrained checkbox
        self.pretrained_check = QCheckBox("Use Pretrained Weights")
        self.pretrained_check.setChecked(True)
        model_layout.addWidget(self.pretrained_check, 3, 0, 1, 2)

        # Freeze backbone checkbox
        self.freeze_backbone_check = QCheckBox("Freeze Backbone")
        model_layout.addWidget(self.freeze_backbone_check, 4, 0, 1, 2)

        layout.addLayout(model_layout)

        # Advanced settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QGridLayout()

        # Dropout rate
        advanced_layout.addWidget(QLabel("Dropout Rate:"), 0, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setSingleStep(0.1)
        self.dropout_spin.setValue(0.5)
        advanced_layout.addWidget(self.dropout_spin, 0, 1)

        # Label smoothing (for EfficientNet)
        advanced_layout.addWidget(QLabel("Label Smoothing:"), 1, 0)
        self.label_smoothing_spin = QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 0.5)
        self.label_smoothing_spin.setSingleStep(0.05)
        self.label_smoothing_spin.setValue(0.1)
        self.label_smoothing_spin.setEnabled(False)
        advanced_layout.addWidget(self.label_smoothing_spin, 1, 1)

        # Stochastic depth (for EfficientNet)
        advanced_layout.addWidget(QLabel("Stochastic Depth:"), 2, 0)
        self.stochastic_depth_spin = QDoubleSpinBox()
        self.stochastic_depth_spin.setRange(0.0, 0.5)
        self.stochastic_depth_spin.setSingleStep(0.05)
        self.stochastic_depth_spin.setValue(0.2)
        self.stochastic_depth_spin.setEnabled(False)
        advanced_layout.addWidget(self.stochastic_depth_spin, 2, 1)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # Model info display
        info_group = QGroupBox("Model Information")
        info_layout = QGridLayout()

        info_layout.addWidget(QLabel("Architecture:"), 0, 0)
        self.architecture_label = QLabel("ResNet-50")
        info_layout.addWidget(self.architecture_label, 0, 1)

        info_layout.addWidget(QLabel("Parameters:"), 1, 0)
        self.parameters_label = QLabel("~25M")
        info_layout.addWidget(self.parameters_label, 1, 1)

        info_layout.addWidget(QLabel("Input Size:"), 2, 0)
        self.input_size_label = QLabel("224x224")
        info_layout.addWidget(self.input_size_label, 2, 1)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset")
        self.apply_button = QPushButton("Apply")
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.apply_button)
        layout.addLayout(button_layout)

        layout.addStretch()
        self.setLayout(layout)

    def setup_connections(self):
        """Setup signal-slot connections."""
        self.model_type_combo.currentIndexChanged.connect(self.on_model_type_changed)
        self.model_variant_combo.currentIndexChanged.connect(self.on_model_variant_changed)
        self.num_classes_spin.valueChanged.connect(self.on_config_changed)
        self.pretrained_check.stateChanged.connect(self.on_config_changed)
        self.freeze_backbone_check.stateChanged.connect(self.on_config_changed)
        self.dropout_spin.valueChanged.connect(self.on_config_changed)
        self.label_smoothing_spin.valueChanged.connect(self.on_config_changed)
        self.stochastic_depth_spin.valueChanged.connect(self.on_config_changed)
        self.reset_button.clicked.connect(self.reset_configuration)
        self.apply_button.clicked.connect(self.apply_configuration)

    def on_model_type_changed(self):
        """Handle model type change."""
        self.update_model_variants()
        self.update_model_info()
        self.update_advanced_settings()
        self.on_config_changed()

    def on_model_variant_changed(self):
        """Handle model variant change."""
        self.update_model_info()
        self.on_config_changed()

    def update_model_variants(self):
        """Update model variants based on selected type."""
        model_type = self.model_type_combo.currentText()
        self.model_variant_combo.clear()

        if model_type == "ResNet":
            self.model_variant_combo.addItems([
                "ResNet-18", "ResNet-34", "ResNet-50",
                "ResNet-101", "ResNet-152"
            ])
            self.model_variant_combo.setCurrentText("ResNet-50")
        elif model_type == "VGG":
            self.model_variant_combo.addItems([
                "VGG-11", "VGG-13", "VGG-16", "VGG-19"
            ])
            self.model_variant_combo.setCurrentText("VGG-16")
        elif model_type == "EfficientNet":
            self.model_variant_combo.addItems([
                "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2",
                "EfficientNet-B3", "EfficientNet-B4", "EfficientNet-B5",
                "EfficientNet-B6", "EfficientNet-B7"
            ])
            self.model_variant_combo.setCurrentText("EfficientNet-B0")

    def update_model_info(self):
        """Update model information display."""
        model_variant = self.model_variant_combo.currentText()
        self.architecture_label.setText(model_variant)

        # Update parameter count (approximate)
        param_counts = {
            "ResNet-18": "~11M", "ResNet-34": "~21M", "ResNet-50": "~25M",
            "ResNet-101": "~44M", "ResNet-152": "~60M",
            "VGG-11": "~132M", "VGG-13": "~133M", "VGG-16": "~138M", "VGG-19": "~143M",
            "EfficientNet-B0": "~5.3M", "EfficientNet-B1": "~7.8M",
            "EfficientNet-B2": "~9.2M", "EfficientNet-B3": "~12M",
            "EfficientNet-B4": "~19M", "EfficientNet-B5": "~30M",
            "EfficientNet-B6": "~43M", "EfficientNet-B7": "~66M"
        }
        self.parameters_label.setText(param_counts.get(model_variant, "Unknown"))

        # Update input size
        if "EfficientNet" in model_variant:
            input_sizes = {
                "EfficientNet-B0": "224x224", "EfficientNet-B1": "240x240",
                "EfficientNet-B2": "260x260", "EfficientNet-B3": "300x300",
                "EfficientNet-B4": "380x380", "EfficientNet-B5": "456x456",
                "EfficientNet-B6": "528x528", "EfficientNet-B7": "600x600"
            }
            self.input_size_label.setText(input_sizes.get(model_variant, "224x224"))
        else:
            self.input_size_label.setText("224x224")

    def update_advanced_settings(self):
        """Enable/disable advanced settings based on model type."""
        model_type = self.model_type_combo.currentText()

        # Enable label smoothing and stochastic depth for EfficientNet
        if model_type == "EfficientNet":
            self.label_smoothing_spin.setEnabled(True)
            self.stochastic_depth_spin.setEnabled(True)
        else:
            self.label_smoothing_spin.setEnabled(False)
            self.stochastic_depth_spin.setEnabled(False)

    def on_config_changed(self):
        """Handle configuration change."""
        config = self.get_configuration()
        self.config_changed.emit(config)

    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration."""
        model_type = self.model_type_combo.currentText().lower()
        model_variant = self.model_variant_combo.currentText().lower().replace("-", "")

        config = {
            "name": model_type,
            "variant": model_variant,
            "num_classes": self.num_classes_spin.value(),
            "pretrained": self.pretrained_check.isChecked(),
            "freeze_backbone": self.freeze_backbone_check.isChecked(),
            "dropout_rate": self.dropout_spin.value()
        }

        # Add model-specific configurations
        if model_type == "resnet":
            config["resnet_version"] = model_variant
        elif model_type == "vgg":
            config["vgg_version"] = model_variant
            config["batch_norm"] = True  # Always use batch norm
        elif model_type == "efficientnet":
            config["efficientnet_version"] = model_variant.replace("efficientnet", "efficientnet_")
            config["label_smoothing"] = self.label_smoothing_spin.value()
            config["stochastic_depth_prob"] = self.stochastic_depth_spin.value()

        return config

    def set_configuration(self, config: Dict[str, Any]):
        """Set the configuration."""
        # Set model type
        model_name = config.get("name", "resnet")
        for i in range(self.model_type_combo.count()):
            if self.model_type_combo.itemText(i).lower() == model_name:
                self.model_type_combo.setCurrentIndex(i)
                break

        # Update variants
        self.update_model_variants()

        # Set model variant
        variant = config.get("variant", "")
        for i in range(self.model_variant_combo.count()):
            if self.model_variant_combo.itemText(i).lower().replace("-", "") == variant:
                self.model_variant_combo.setCurrentIndex(i)
                break

        # Set other values
        self.num_classes_spin.setValue(config.get("num_classes", 10))
        self.pretrained_check.setChecked(config.get("pretrained", True))
        self.freeze_backbone_check.setChecked(config.get("freeze_backbone", False))
        self.dropout_spin.setValue(config.get("dropout_rate", 0.5))

        # Set model-specific values
        if model_name == "efficientnet":
            self.label_smoothing_spin.setValue(config.get("label_smoothing", 0.1))
            self.stochastic_depth_spin.setValue(config.get("stochastic_depth_prob", 0.2))

        self.update_model_info()
        self.update_advanced_settings()

    def reset_configuration(self):
        """Reset to default configuration."""
        self.set_default_values()

    def set_default_values(self):
        """Set default values."""
        self.model_type_combo.setCurrentIndex(0)
        self.update_model_variants()
        self.model_variant_combo.setCurrentText("ResNet-50")
        self.num_classes_spin.setValue(10)
        self.pretrained_check.setChecked(True)
        self.freeze_backbone_check.setChecked(False)
        self.dropout_spin.setValue(0.5)
        self.label_smoothing_spin.setValue(0.1)
        self.stochastic_depth_spin.setValue(0.2)
        self.update_model_info()
        self.update_advanced_settings()

    def apply_configuration(self):
        """Apply the current configuration."""
        config = self.get_configuration()
        self.config_changed.emit(config)