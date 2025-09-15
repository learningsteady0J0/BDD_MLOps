"""Widget for dataset configuration."""

import numpy as np
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QGridLayout, QLineEdit,
    QFileDialog, QListWidget, QTabWidget, QTextEdit
)
from PyQt5.QtCore import pyqtSignal, Qt, QThread
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DatasetPreviewThread(QThread):
    """Thread for loading dataset preview."""

    preview_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, dataset_name: str):
        super().__init__()
        self.dataset_name = dataset_name

    def run(self):
        """Load dataset preview."""
        try:
            # Simulate loading dataset info
            if self.dataset_name == "CIFAR-10":
                preview_data = {
                    "num_classes": 10,
                    "num_train": 50000,
                    "num_val": 10000,
                    "num_test": 10000,
                    "image_size": (32, 32, 3),
                    "classes": ["airplane", "automobile", "bird", "cat", "deer",
                               "dog", "frog", "horse", "ship", "truck"]
                }
            elif self.dataset_name == "CIFAR-100":
                preview_data = {
                    "num_classes": 100,
                    "num_train": 50000,
                    "num_val": 10000,
                    "num_test": 10000,
                    "image_size": (32, 32, 3),
                    "classes": [f"Class_{i}" for i in range(100)]
                }
            else:
                preview_data = {
                    "num_classes": 10,
                    "num_train": 60000,
                    "num_val": 10000,
                    "num_test": 10000,
                    "image_size": (224, 224, 3),
                    "classes": [f"Class_{i}" for i in range(10)]
                }

            self.preview_ready.emit(preview_data)

        except Exception as e:
            self.error_occurred.emit(str(e))


class DatasetConfigWidget(QGroupBox):
    """Widget for configuring dataset settings."""

    # Signal emitted when configuration changes
    config_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        """Initialize the dataset configuration widget."""
        super().__init__("Dataset Configuration", parent)
        self.preview_thread = None
        self.init_ui()
        self.setup_connections()
        self.set_default_values()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Create tabs
        self.tabs = QTabWidget()

        # Dataset selection tab
        dataset_tab = QWidget()
        dataset_layout = QVBoxLayout(dataset_tab)

        # Dataset selection
        selection_layout = QGridLayout()

        selection_layout.addWidget(QLabel("Dataset:"), 0, 0)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["CIFAR-10", "CIFAR-100", "Custom"])
        selection_layout.addWidget(self.dataset_combo, 0, 1)

        # Data directory
        selection_layout.addWidget(QLabel("Data Directory:"), 1, 0)
        data_dir_layout = QHBoxLayout()
        self.data_dir_edit = QLineEdit("./data")
        self.browse_button = QPushButton("Browse...")
        data_dir_layout.addWidget(self.data_dir_edit)
        data_dir_layout.addWidget(self.browse_button)
        selection_layout.addLayout(data_dir_layout, 1, 1)

        # Batch size
        selection_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(32)
        selection_layout.addWidget(self.batch_size_spin, 2, 1)

        # Number of workers
        selection_layout.addWidget(QLabel("Num Workers:"), 3, 0)
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 16)
        self.num_workers_spin.setValue(4)
        selection_layout.addWidget(self.num_workers_spin, 3, 1)

        # Validation split
        selection_layout.addWidget(QLabel("Validation Split:"), 4, 0)
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.0, 0.5)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.2)
        self.val_split_spin.setSuffix(" %")
        selection_layout.addWidget(self.val_split_spin, 4, 1)

        dataset_layout.addLayout(selection_layout)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()

        self.pin_memory_check = QCheckBox("Pin Memory (GPU)")
        self.pin_memory_check.setChecked(True)
        options_layout.addWidget(self.pin_memory_check)

        self.persistent_workers_check = QCheckBox("Persistent Workers")
        self.persistent_workers_check.setChecked(True)
        options_layout.addWidget(self.persistent_workers_check)

        self.shuffle_train_check = QCheckBox("Shuffle Training Data")
        self.shuffle_train_check.setChecked(True)
        options_layout.addWidget(self.shuffle_train_check)

        self.drop_last_check = QCheckBox("Drop Last Batch")
        options_layout.addWidget(self.drop_last_check)

        options_group.setLayout(options_layout)
        dataset_layout.addWidget(options_group)

        dataset_layout.addStretch()

        # Augmentation tab
        augmentation_tab = QWidget()
        augmentation_layout = QVBoxLayout(augmentation_tab)

        # Augmentation presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.augmentation_preset_combo = QComboBox()
        self.augmentation_preset_combo.addItems([
            "None", "Basic", "Standard", "Advanced", "Custom"
        ])
        self.augmentation_preset_combo.setCurrentText("Standard")
        preset_layout.addWidget(self.augmentation_preset_combo)
        preset_layout.addStretch()
        augmentation_layout.addLayout(preset_layout)

        # Augmentation options
        aug_options_group = QGroupBox("Augmentation Options")
        aug_options_layout = QVBoxLayout()

        self.random_crop_check = QCheckBox("Random Crop")
        self.random_crop_check.setChecked(True)
        aug_options_layout.addWidget(self.random_crop_check)

        self.random_flip_check = QCheckBox("Random Horizontal Flip")
        self.random_flip_check.setChecked(True)
        aug_options_layout.addWidget(self.random_flip_check)

        self.color_jitter_check = QCheckBox("Color Jitter")
        self.color_jitter_check.setChecked(True)
        aug_options_layout.addWidget(self.color_jitter_check)

        self.random_rotation_check = QCheckBox("Random Rotation")
        aug_options_layout.addWidget(self.random_rotation_check)

        self.normalize_check = QCheckBox("Normalize (ImageNet stats)")
        self.normalize_check.setChecked(True)
        aug_options_layout.addWidget(self.normalize_check)

        self.mixup_check = QCheckBox("MixUp")
        aug_options_layout.addWidget(self.mixup_check)

        self.cutmix_check = QCheckBox("CutMix")
        aug_options_layout.addWidget(self.cutmix_check)

        aug_options_group.setLayout(aug_options_layout)
        augmentation_layout.addWidget(aug_options_group)

        # Augmentation parameters
        aug_params_group = QGroupBox("Parameters")
        aug_params_layout = QGridLayout()

        aug_params_layout.addWidget(QLabel("Rotation Degrees:"), 0, 0)
        self.rotation_degrees_spin = QSpinBox()
        self.rotation_degrees_spin.setRange(0, 180)
        self.rotation_degrees_spin.setValue(15)
        aug_params_layout.addWidget(self.rotation_degrees_spin, 0, 1)

        aug_params_layout.addWidget(QLabel("Jitter Brightness:"), 1, 0)
        self.jitter_brightness_spin = QDoubleSpinBox()
        self.jitter_brightness_spin.setRange(0.0, 1.0)
        self.jitter_brightness_spin.setSingleStep(0.1)
        self.jitter_brightness_spin.setValue(0.2)
        aug_params_layout.addWidget(self.jitter_brightness_spin, 1, 1)

        aug_params_layout.addWidget(QLabel("MixUp Alpha:"), 2, 0)
        self.mixup_alpha_spin = QDoubleSpinBox()
        self.mixup_alpha_spin.setRange(0.0, 2.0)
        self.mixup_alpha_spin.setSingleStep(0.1)
        self.mixup_alpha_spin.setValue(0.2)
        aug_params_layout.addWidget(self.mixup_alpha_spin, 2, 1)

        aug_params_group.setLayout(aug_params_layout)
        augmentation_layout.addWidget(aug_params_group)

        augmentation_layout.addStretch()

        # Preview tab
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)

        # Dataset info
        info_group = QGroupBox("Dataset Information")
        info_layout = QGridLayout()

        info_layout.addWidget(QLabel("Classes:"), 0, 0)
        self.classes_label = QLabel("10")
        info_layout.addWidget(self.classes_label, 0, 1)

        info_layout.addWidget(QLabel("Training Samples:"), 1, 0)
        self.train_samples_label = QLabel("50,000")
        info_layout.addWidget(self.train_samples_label, 1, 1)

        info_layout.addWidget(QLabel("Validation Samples:"), 2, 0)
        self.val_samples_label = QLabel("10,000")
        info_layout.addWidget(self.val_samples_label, 2, 1)

        info_layout.addWidget(QLabel("Test Samples:"), 3, 0)
        self.test_samples_label = QLabel("10,000")
        info_layout.addWidget(self.test_samples_label, 3, 1)

        info_layout.addWidget(QLabel("Image Size:"), 4, 0)
        self.image_size_label = QLabel("32x32x3")
        info_layout.addWidget(self.image_size_label, 4, 1)

        info_group.setLayout(info_layout)
        preview_layout.addWidget(info_group)

        # Sample images preview
        preview_group = QGroupBox("Sample Images")
        sample_layout = QVBoxLayout()

        # Create matplotlib figure for preview
        self.preview_figure = Figure(figsize=(8, 4))
        self.preview_canvas = FigureCanvas(self.preview_figure)
        sample_layout.addWidget(self.preview_canvas)

        self.load_preview_button = QPushButton("Load Preview")
        sample_layout.addWidget(self.load_preview_button)

        preview_group.setLayout(sample_layout)
        preview_layout.addWidget(preview_group)

        # Add tabs
        self.tabs.addTab(dataset_tab, "Dataset")
        self.tabs.addTab(augmentation_tab, "Augmentation")
        self.tabs.addTab(preview_tab, "Preview")

        layout.addWidget(self.tabs)

        # Buttons
        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset")
        self.apply_button = QPushButton("Apply")
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.apply_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def setup_connections(self):
        """Setup signal-slot connections."""
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)
        self.browse_button.clicked.connect(self.browse_data_directory)
        self.batch_size_spin.valueChanged.connect(self.on_config_changed)
        self.num_workers_spin.valueChanged.connect(self.on_config_changed)
        self.val_split_spin.valueChanged.connect(self.on_config_changed)

        # Augmentation preset
        self.augmentation_preset_combo.currentIndexChanged.connect(self.on_augmentation_preset_changed)

        # All checkboxes
        for check in [self.pin_memory_check, self.persistent_workers_check,
                     self.shuffle_train_check, self.drop_last_check,
                     self.random_crop_check, self.random_flip_check,
                     self.color_jitter_check, self.random_rotation_check,
                     self.normalize_check, self.mixup_check, self.cutmix_check]:
            check.stateChanged.connect(self.on_config_changed)

        # Augmentation parameters
        self.rotation_degrees_spin.valueChanged.connect(self.on_config_changed)
        self.jitter_brightness_spin.valueChanged.connect(self.on_config_changed)
        self.mixup_alpha_spin.valueChanged.connect(self.on_config_changed)

        # Buttons
        self.reset_button.clicked.connect(self.reset_configuration)
        self.apply_button.clicked.connect(self.apply_configuration)
        self.load_preview_button.clicked.connect(self.load_dataset_preview)

    def on_dataset_changed(self):
        """Handle dataset selection change."""
        dataset = self.dataset_combo.currentText()

        # Update dataset info
        self.update_dataset_info(dataset)

        # Emit configuration change
        self.on_config_changed()

    def update_dataset_info(self, dataset: str):
        """Update dataset information display."""
        if dataset == "CIFAR-10":
            self.classes_label.setText("10")
            self.train_samples_label.setText("50,000")
            self.val_samples_label.setText("10,000")
            self.test_samples_label.setText("10,000")
            self.image_size_label.setText("32x32x3")
        elif dataset == "CIFAR-100":
            self.classes_label.setText("100")
            self.train_samples_label.setText("50,000")
            self.val_samples_label.setText("10,000")
            self.test_samples_label.setText("10,000")
            self.image_size_label.setText("32x32x3")
        else:
            self.classes_label.setText("Custom")
            self.train_samples_label.setText("Custom")
            self.val_samples_label.setText("Custom")
            self.test_samples_label.setText("Custom")
            self.image_size_label.setText("Custom")

    def on_augmentation_preset_changed(self):
        """Handle augmentation preset change."""
        preset = self.augmentation_preset_combo.currentText()

        if preset == "None":
            self.set_augmentation_options(crop=False, flip=False, jitter=False,
                                        rotation=False, normalize=True,
                                        mixup=False, cutmix=False)
        elif preset == "Basic":
            self.set_augmentation_options(crop=True, flip=True, jitter=False,
                                        rotation=False, normalize=True,
                                        mixup=False, cutmix=False)
        elif preset == "Standard":
            self.set_augmentation_options(crop=True, flip=True, jitter=True,
                                        rotation=False, normalize=True,
                                        mixup=False, cutmix=False)
        elif preset == "Advanced":
            self.set_augmentation_options(crop=True, flip=True, jitter=True,
                                        rotation=True, normalize=True,
                                        mixup=True, cutmix=False)

        self.on_config_changed()

    def set_augmentation_options(self, crop=True, flip=True, jitter=False,
                                rotation=False, normalize=True,
                                mixup=False, cutmix=False):
        """Set augmentation options."""
        self.random_crop_check.setChecked(crop)
        self.random_flip_check.setChecked(flip)
        self.color_jitter_check.setChecked(jitter)
        self.random_rotation_check.setChecked(rotation)
        self.normalize_check.setChecked(normalize)
        self.mixup_check.setChecked(mixup)
        self.cutmix_check.setChecked(cutmix)

    def browse_data_directory(self):
        """Browse for data directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            self.data_dir_edit.text()
        )
        if directory:
            self.data_dir_edit.setText(directory)
            self.on_config_changed()

    def load_dataset_preview(self):
        """Load dataset preview."""
        dataset = self.dataset_combo.currentText()

        # Start preview thread
        self.preview_thread = DatasetPreviewThread(dataset)
        self.preview_thread.preview_ready.connect(self.display_preview)
        self.preview_thread.error_occurred.connect(self.on_preview_error)
        self.preview_thread.start()

        self.load_preview_button.setEnabled(False)
        self.load_preview_button.setText("Loading...")

    def display_preview(self, preview_data: Dict[str, Any]):
        """Display dataset preview."""
        # Update info
        self.classes_label.setText(str(preview_data["num_classes"]))
        self.train_samples_label.setText(f"{preview_data['num_train']:,}")
        self.val_samples_label.setText(f"{preview_data['num_val']:,}")
        self.test_samples_label.setText(f"{preview_data['num_test']:,}")

        size = preview_data["image_size"]
        self.image_size_label.setText(f"{size[0]}x{size[1]}x{size[2]}")

        # Create sample preview
        self.preview_figure.clear()

        # Create grid of sample images (simulated)
        rows, cols = 2, 4
        for i in range(rows * cols):
            ax = self.preview_figure.add_subplot(rows, cols, i + 1)

            # Generate random image for demonstration
            if size[0] == 32:  # CIFAR
                img = np.random.rand(32, 32, 3)
            else:
                img = np.random.rand(64, 64, 3)

            ax.imshow(img)
            ax.axis('off')
            if i < len(preview_data["classes"]):
                ax.set_title(preview_data["classes"][i], fontsize=8)

        self.preview_figure.tight_layout()
        self.preview_canvas.draw()

        # Re-enable button
        self.load_preview_button.setEnabled(True)
        self.load_preview_button.setText("Load Preview")

    def on_preview_error(self, error: str):
        """Handle preview error."""
        self.load_preview_button.setEnabled(True)
        self.load_preview_button.setText("Load Preview")
        # Could show error message here

    def on_config_changed(self):
        """Handle configuration change."""
        config = self.get_configuration()
        self.config_changed.emit(config)

    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration."""
        dataset = self.dataset_combo.currentText()

        config = {
            "name": dataset.lower().replace("-", ""),
            "data_dir": self.data_dir_edit.text(),
            "batch_size": self.batch_size_spin.value(),
            "num_workers": self.num_workers_spin.value(),
            "val_split": self.val_split_spin.value(),
            "pin_memory": self.pin_memory_check.isChecked(),
            "persistent_workers": self.persistent_workers_check.isChecked(),
            "shuffle_train": self.shuffle_train_check.isChecked(),
            "drop_last": self.drop_last_check.isChecked(),
            "augmentation": {
                "preset": self.augmentation_preset_combo.currentText(),
                "random_crop": self.random_crop_check.isChecked(),
                "random_flip": self.random_flip_check.isChecked(),
                "color_jitter": self.color_jitter_check.isChecked(),
                "random_rotation": self.random_rotation_check.isChecked(),
                "normalize": self.normalize_check.isChecked(),
                "mixup": self.mixup_check.isChecked(),
                "cutmix": self.cutmix_check.isChecked(),
                "rotation_degrees": self.rotation_degrees_spin.value(),
                "jitter_brightness": self.jitter_brightness_spin.value(),
                "mixup_alpha": self.mixup_alpha_spin.value()
            }
        }

        return config

    def set_configuration(self, config: Dict[str, Any]):
        """Set the configuration."""
        # Set dataset
        dataset_name = config.get("name", "cifar10")
        dataset_map = {
            "cifar10": "CIFAR-10",
            "cifar100": "CIFAR-100",
            "custom": "Custom"
        }
        self.dataset_combo.setCurrentText(dataset_map.get(dataset_name, "CIFAR-10"))

        # Set other values
        self.data_dir_edit.setText(config.get("data_dir", "./data"))
        self.batch_size_spin.setValue(config.get("batch_size", 32))
        self.num_workers_spin.setValue(config.get("num_workers", 4))
        self.val_split_spin.setValue(config.get("val_split", 0.2))

        # Set options
        self.pin_memory_check.setChecked(config.get("pin_memory", True))
        self.persistent_workers_check.setChecked(config.get("persistent_workers", True))
        self.shuffle_train_check.setChecked(config.get("shuffle_train", True))
        self.drop_last_check.setChecked(config.get("drop_last", False))

        # Set augmentation
        aug_config = config.get("augmentation", {})
        self.augmentation_preset_combo.setCurrentText(aug_config.get("preset", "Standard"))

        if aug_config.get("preset") == "Custom":
            self.random_crop_check.setChecked(aug_config.get("random_crop", True))
            self.random_flip_check.setChecked(aug_config.get("random_flip", True))
            self.color_jitter_check.setChecked(aug_config.get("color_jitter", True))
            self.random_rotation_check.setChecked(aug_config.get("random_rotation", False))
            self.normalize_check.setChecked(aug_config.get("normalize", True))
            self.mixup_check.setChecked(aug_config.get("mixup", False))
            self.cutmix_check.setChecked(aug_config.get("cutmix", False))

        self.rotation_degrees_spin.setValue(aug_config.get("rotation_degrees", 15))
        self.jitter_brightness_spin.setValue(aug_config.get("jitter_brightness", 0.2))
        self.mixup_alpha_spin.setValue(aug_config.get("mixup_alpha", 0.2))

        # Update info
        self.update_dataset_info(self.dataset_combo.currentText())

    def reset_configuration(self):
        """Reset to default configuration."""
        self.set_default_values()

    def set_default_values(self):
        """Set default values."""
        self.dataset_combo.setCurrentIndex(0)
        self.data_dir_edit.setText("./data")
        self.batch_size_spin.setValue(32)
        self.num_workers_spin.setValue(4)
        self.val_split_spin.setValue(0.2)

        self.pin_memory_check.setChecked(True)
        self.persistent_workers_check.setChecked(True)
        self.shuffle_train_check.setChecked(True)
        self.drop_last_check.setChecked(False)

        self.augmentation_preset_combo.setCurrentText("Standard")
        self.on_augmentation_preset_changed()

        self.update_dataset_info("CIFAR-10")

    def apply_configuration(self):
        """Apply the current configuration."""
        config = self.get_configuration()
        self.config_changed.emit(config)