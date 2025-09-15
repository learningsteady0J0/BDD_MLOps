"""Widget for training control and monitoring."""

from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QGridLayout, QProgressBar,
    QTextEdit, QTabWidget, QLineEdit, QSlider
)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor


class TrainingControlWidget(QGroupBox):
    """Widget for controlling and monitoring training."""

    # Signals
    start_training = pyqtSignal()
    stop_training = pyqtSignal()
    pause_training = pyqtSignal()
    resume_training = pyqtSignal()
    config_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        """Initialize the training control widget."""
        super().__init__("Training Control", parent)
        self.is_training = False
        self.is_paused = False
        self.init_ui()
        self.setup_connections()
        self.set_default_values()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Create tabs
        self.tabs = QTabWidget()

        # Training Parameters Tab
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)

        # Optimizer settings
        optimizer_group = QGroupBox("Optimizer Settings")
        optimizer_layout = QGridLayout()

        optimizer_layout.addWidget(QLabel("Optimizer:"), 0, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "AdamW", "SGD", "RMSprop"])
        optimizer_layout.addWidget(self.optimizer_combo, 0, 1)

        optimizer_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0000001, 1.0)
        self.learning_rate_spin.setDecimals(7)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setValue(0.001)
        optimizer_layout.addWidget(self.learning_rate_spin, 1, 1)

        optimizer_layout.addWidget(QLabel("Weight Decay:"), 2, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(0.0001)
        optimizer_layout.addWidget(self.weight_decay_spin, 2, 1)

        optimizer_layout.addWidget(QLabel("Momentum (SGD):"), 3, 0)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setSingleStep(0.1)
        self.momentum_spin.setValue(0.9)
        optimizer_layout.addWidget(self.momentum_spin, 3, 1)

        optimizer_group.setLayout(optimizer_layout)
        params_layout.addWidget(optimizer_group)

        # Scheduler settings
        scheduler_group = QGroupBox("Learning Rate Scheduler")
        scheduler_layout = QGridLayout()

        scheduler_layout.addWidget(QLabel("Scheduler:"), 0, 0)
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems([
            "None", "StepLR", "CosineAnnealing", "ReduceLROnPlateau",
            "ExponentialLR", "OneCycleLR"
        ])
        scheduler_layout.addWidget(self.scheduler_combo, 0, 1)

        scheduler_layout.addWidget(QLabel("Step Size:"), 1, 0)
        self.step_size_spin = QSpinBox()
        self.step_size_spin.setRange(1, 100)
        self.step_size_spin.setValue(10)
        scheduler_layout.addWidget(self.step_size_spin, 1, 1)

        scheduler_layout.addWidget(QLabel("Gamma:"), 2, 0)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 1.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(0.1)
        scheduler_layout.addWidget(self.gamma_spin, 2, 1)

        scheduler_group.setLayout(scheduler_layout)
        params_layout.addWidget(scheduler_group)

        # Training settings
        training_group = QGroupBox("Training Settings")
        training_layout = QGridLayout()

        training_layout.addWidget(QLabel("Max Epochs:"), 0, 0)
        self.max_epochs_spin = QSpinBox()
        self.max_epochs_spin.setRange(1, 1000)
        self.max_epochs_spin.setValue(100)
        training_layout.addWidget(self.max_epochs_spin, 0, 1)

        training_layout.addWidget(QLabel("Gradient Clip:"), 1, 0)
        self.gradient_clip_spin = QDoubleSpinBox()
        self.gradient_clip_spin.setRange(0.0, 10.0)
        self.gradient_clip_spin.setSingleStep(0.5)
        self.gradient_clip_spin.setValue(1.0)
        training_layout.addWidget(self.gradient_clip_spin, 1, 1)

        training_layout.addWidget(QLabel("Accumulate Batches:"), 2, 0)
        self.accumulate_batches_spin = QSpinBox()
        self.accumulate_batches_spin.setRange(1, 32)
        self.accumulate_batches_spin.setValue(1)
        training_layout.addWidget(self.accumulate_batches_spin, 2, 1)

        training_layout.addWidget(QLabel("Precision:"), 3, 0)
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["16", "32", "bf16"])
        self.precision_combo.setCurrentText("16")
        training_layout.addWidget(self.precision_combo, 3, 1)

        training_group.setLayout(training_layout)
        params_layout.addWidget(training_group)

        params_layout.addStretch()

        # Control Tab
        control_tab = QWidget()
        control_layout = QVBoxLayout(control_tab)

        # Training controls
        control_buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)

        control_buttons_layout.addWidget(self.start_button)
        control_buttons_layout.addWidget(self.pause_button)
        control_buttons_layout.addWidget(self.stop_button)
        control_layout.addLayout(control_buttons_layout)

        # Progress display
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()

        # Overall progress
        overall_layout = QHBoxLayout()
        overall_layout.addWidget(QLabel("Overall:"))
        self.overall_progress = QProgressBar()
        self.overall_progress.setTextVisible(True)
        overall_layout.addWidget(self.overall_progress)
        progress_layout.addLayout(overall_layout)

        # Epoch progress
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("Epoch:"))
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setTextVisible(True)
        epoch_layout.addWidget(self.epoch_progress)
        progress_layout.addLayout(epoch_layout)

        # Current status
        status_layout = QGridLayout()

        status_layout.addWidget(QLabel("Status:"), 0, 0)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label, 0, 1)

        status_layout.addWidget(QLabel("Current Epoch:"), 1, 0)
        self.current_epoch_label = QLabel("0 / 0")
        status_layout.addWidget(self.current_epoch_label, 1, 1)

        status_layout.addWidget(QLabel("Batch:"), 2, 0)
        self.current_batch_label = QLabel("0 / 0")
        status_layout.addWidget(self.current_batch_label, 2, 1)

        status_layout.addWidget(QLabel("Time Elapsed:"), 3, 0)
        self.time_elapsed_label = QLabel("00:00:00")
        status_layout.addWidget(self.time_elapsed_label, 3, 1)

        status_layout.addWidget(QLabel("Time Remaining:"), 4, 0)
        self.time_remaining_label = QLabel("00:00:00")
        status_layout.addWidget(self.time_remaining_label, 4, 1)

        progress_layout.addLayout(status_layout)

        progress_group.setLayout(progress_layout)
        control_layout.addWidget(progress_group)

        # Current metrics
        metrics_group = QGroupBox("Current Metrics")
        metrics_layout = QGridLayout()

        metrics_layout.addWidget(QLabel("Train Loss:"), 0, 0)
        self.train_loss_label = QLabel("0.0000")
        self.train_loss_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        metrics_layout.addWidget(self.train_loss_label, 0, 1)

        metrics_layout.addWidget(QLabel("Val Loss:"), 1, 0)
        self.val_loss_label = QLabel("0.0000")
        self.val_loss_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        metrics_layout.addWidget(self.val_loss_label, 1, 1)

        metrics_layout.addWidget(QLabel("Train Acc:"), 2, 0)
        self.train_acc_label = QLabel("0.00%")
        self.train_acc_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        metrics_layout.addWidget(self.train_acc_label, 2, 1)

        metrics_layout.addWidget(QLabel("Val Acc:"), 3, 0)
        self.val_acc_label = QLabel("0.00%")
        self.val_acc_label.setStyleSheet("color: #9C27B0; font-weight: bold;")
        metrics_layout.addWidget(self.val_acc_label, 3, 1)

        metrics_layout.addWidget(QLabel("Learning Rate:"), 4, 0)
        self.lr_label = QLabel("0.0000")
        metrics_layout.addWidget(self.lr_label, 4, 1)

        metrics_group.setLayout(metrics_layout)
        control_layout.addWidget(metrics_group)

        control_layout.addStretch()

        # Early Stopping Tab
        early_stop_tab = QWidget()
        early_stop_layout = QVBoxLayout(early_stop_tab)

        early_stop_group = QGroupBox("Early Stopping Settings")
        early_stop_grid = QGridLayout()

        self.early_stopping_check = QCheckBox("Enable Early Stopping")
        early_stop_grid.addWidget(self.early_stopping_check, 0, 0, 1, 2)

        early_stop_grid.addWidget(QLabel("Monitor Metric:"), 1, 0)
        self.monitor_metric_combo = QComboBox()
        self.monitor_metric_combo.addItems(["val_loss", "val_acc", "train_loss"])
        early_stop_grid.addWidget(self.monitor_metric_combo, 1, 1)

        early_stop_grid.addWidget(QLabel("Patience:"), 2, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(10)
        early_stop_grid.addWidget(self.patience_spin, 2, 1)

        early_stop_grid.addWidget(QLabel("Min Delta:"), 3, 0)
        self.min_delta_spin = QDoubleSpinBox()
        self.min_delta_spin.setRange(0.0, 1.0)
        self.min_delta_spin.setDecimals(6)
        self.min_delta_spin.setSingleStep(0.0001)
        self.min_delta_spin.setValue(0.0001)
        early_stop_grid.addWidget(self.min_delta_spin, 3, 1)

        early_stop_grid.addWidget(QLabel("Mode:"), 4, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["min", "max"])
        early_stop_grid.addWidget(self.mode_combo, 4, 1)

        early_stop_group.setLayout(early_stop_grid)
        early_stop_layout.addWidget(early_stop_group)

        # Checkpointing settings
        checkpoint_group = QGroupBox("Checkpointing")
        checkpoint_grid = QGridLayout()

        self.save_checkpoint_check = QCheckBox("Save Checkpoints")
        self.save_checkpoint_check.setChecked(True)
        checkpoint_grid.addWidget(self.save_checkpoint_check, 0, 0, 1, 2)

        checkpoint_grid.addWidget(QLabel("Save Top K:"), 1, 0)
        self.save_top_k_spin = QSpinBox()
        self.save_top_k_spin.setRange(1, 10)
        self.save_top_k_spin.setValue(3)
        checkpoint_grid.addWidget(self.save_top_k_spin, 1, 1)

        checkpoint_grid.addWidget(QLabel("Save Every N Epochs:"), 2, 0)
        self.save_every_n_spin = QSpinBox()
        self.save_every_n_spin.setRange(1, 100)
        self.save_every_n_spin.setValue(1)
        checkpoint_grid.addWidget(self.save_every_n_spin, 2, 1)

        self.save_last_check = QCheckBox("Save Last Checkpoint")
        self.save_last_check.setChecked(True)
        checkpoint_grid.addWidget(self.save_last_check, 3, 0, 1, 2)

        checkpoint_group.setLayout(checkpoint_grid)
        early_stop_layout.addWidget(checkpoint_group)

        early_stop_layout.addStretch()

        # Add tabs
        self.tabs.addTab(params_tab, "Parameters")
        self.tabs.addTab(control_tab, "Control")
        self.tabs.addTab(early_stop_tab, "Early Stopping")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def setup_connections(self):
        """Setup signal-slot connections."""
        # Control buttons
        self.start_button.clicked.connect(self.on_start_clicked)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.pause_button.clicked.connect(self.on_pause_clicked)

        # Configuration changes
        self.optimizer_combo.currentIndexChanged.connect(self.on_optimizer_changed)
        self.scheduler_combo.currentIndexChanged.connect(self.on_scheduler_changed)

        # All value changes
        for widget in [
            self.learning_rate_spin, self.weight_decay_spin, self.momentum_spin,
            self.step_size_spin, self.gamma_spin, self.max_epochs_spin,
            self.gradient_clip_spin, self.accumulate_batches_spin,
            self.patience_spin, self.min_delta_spin, self.save_top_k_spin,
            self.save_every_n_spin
        ]:
            widget.valueChanged.connect(self.on_config_changed)

        # Combo boxes
        for combo in [self.precision_combo, self.monitor_metric_combo, self.mode_combo]:
            combo.currentIndexChanged.connect(self.on_config_changed)

        # Checkboxes
        for check in [self.early_stopping_check, self.save_checkpoint_check, self.save_last_check]:
            check.stateChanged.connect(self.on_config_changed)

    def on_optimizer_changed(self):
        """Handle optimizer selection change."""
        optimizer = self.optimizer_combo.currentText()
        # Enable/disable momentum for SGD
        self.momentum_spin.setEnabled(optimizer == "SGD")
        self.on_config_changed()

    def on_scheduler_changed(self):
        """Handle scheduler selection change."""
        scheduler = self.scheduler_combo.currentText()
        # Enable/disable scheduler-specific parameters
        self.step_size_spin.setEnabled(scheduler in ["StepLR", "CosineAnnealing"])
        self.gamma_spin.setEnabled(scheduler in ["StepLR", "ExponentialLR"])
        self.on_config_changed()

    def on_start_clicked(self):
        """Handle start button click."""
        if not self.is_training:
            self.start_training.emit()
        elif self.is_paused:
            self.resume_training.emit()

    def on_stop_clicked(self):
        """Handle stop button click."""
        self.stop_training.emit()

    def on_pause_clicked(self):
        """Handle pause button click."""
        if not self.is_paused:
            self.pause_training.emit()
            self.pause_button.setText("Resume")
            self.is_paused = True
        else:
            self.resume_training.emit()
            self.pause_button.setText("Pause")
            self.is_paused = False

    def set_training_active(self, active: bool):
        """Set training active state."""
        self.is_training = active
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(active)
        self.pause_button.setEnabled(active)

        if active:
            self.status_label.setText("Training Active")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.tabs.setCurrentIndex(1)  # Switch to Control tab
        else:
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("font-weight: bold;")
            self.is_paused = False
            self.pause_button.setText("Pause")

    def update_progress(self, progress_data: Dict[str, Any]):
        """Update training progress display."""
        # Update progress bars
        if "overall_progress" in progress_data:
            self.overall_progress.setValue(int(progress_data["overall_progress"]))

        if "epoch_progress" in progress_data:
            self.epoch_progress.setValue(int(progress_data["epoch_progress"]))

        # Update labels
        if "current_epoch" in progress_data:
            self.current_epoch_label.setText(
                f"{progress_data['current_epoch']} / {progress_data.get('total_epochs', 0)}"
            )

        if "current_batch" in progress_data:
            self.current_batch_label.setText(
                f"{progress_data['current_batch']} / {progress_data.get('total_batches', 0)}"
            )

        if "time_elapsed" in progress_data:
            self.time_elapsed_label.setText(progress_data["time_elapsed"])

        if "time_remaining" in progress_data:
            self.time_remaining_label.setText(progress_data["time_remaining"])

        # Update metrics
        if "train_loss" in progress_data:
            self.train_loss_label.setText(f"{progress_data['train_loss']:.4f}")

        if "val_loss" in progress_data:
            self.val_loss_label.setText(f"{progress_data['val_loss']:.4f}")

        if "train_acc" in progress_data:
            self.train_acc_label.setText(f"{progress_data['train_acc']:.2f}%")

        if "val_acc" in progress_data:
            self.val_acc_label.setText(f"{progress_data['val_acc']:.2f}%")

        if "learning_rate" in progress_data:
            self.lr_label.setText(f"{progress_data['learning_rate']:.6f}")

    def on_config_changed(self):
        """Handle configuration change."""
        config = self.get_configuration()
        self.config_changed.emit(config)

    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration."""
        config = {
            "optimizer": self.optimizer_combo.currentText().lower(),
            "learning_rate": self.learning_rate_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "max_epochs": self.max_epochs_spin.value(),
            "gradient_clip_val": self.gradient_clip_spin.value(),
            "accumulate_grad_batches": self.accumulate_batches_spin.value(),
            "precision": int(self.precision_combo.currentText()) if self.precision_combo.currentText() != "bf16" else "bf16",
            "scheduler": {
                "type": self.scheduler_combo.currentText(),
                "step_size": self.step_size_spin.value(),
                "gamma": self.gamma_spin.value()
            }
        }

        # Add optimizer-specific parameters
        if config["optimizer"] == "sgd":
            config["momentum"] = self.momentum_spin.value()

        # Early stopping
        if self.early_stopping_check.isChecked():
            config["early_stopping"] = {
                "monitor": self.monitor_metric_combo.currentText(),
                "patience": self.patience_spin.value(),
                "min_delta": self.min_delta_spin.value(),
                "mode": self.mode_combo.currentText()
            }

        # Checkpointing
        config["checkpoint"] = {
            "save_top_k": self.save_top_k_spin.value() if self.save_checkpoint_check.isChecked() else 0,
            "save_every_n_epochs": self.save_every_n_spin.value(),
            "save_last": self.save_last_check.isChecked()
        }

        return config

    def set_configuration(self, config: Dict[str, Any]):
        """Set the configuration."""
        # Set optimizer
        optimizer = config.get("optimizer", "adamw")
        for i in range(self.optimizer_combo.count()):
            if self.optimizer_combo.itemText(i).lower() == optimizer:
                self.optimizer_combo.setCurrentIndex(i)
                break

        # Set values
        self.learning_rate_spin.setValue(config.get("learning_rate", 0.001))
        self.weight_decay_spin.setValue(config.get("weight_decay", 0.0001))
        self.max_epochs_spin.setValue(config.get("max_epochs", 100))
        self.gradient_clip_spin.setValue(config.get("gradient_clip_val", 1.0))
        self.accumulate_batches_spin.setValue(config.get("accumulate_grad_batches", 1))

        if optimizer == "sgd":
            self.momentum_spin.setValue(config.get("momentum", 0.9))

        # Set precision
        precision = str(config.get("precision", 16))
        self.precision_combo.setCurrentText(precision)

        # Set scheduler
        scheduler_config = config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "None")
        for i in range(self.scheduler_combo.count()):
            if self.scheduler_combo.itemText(i) == scheduler_type:
                self.scheduler_combo.setCurrentIndex(i)
                break

        self.step_size_spin.setValue(scheduler_config.get("step_size", 10))
        self.gamma_spin.setValue(scheduler_config.get("gamma", 0.1))

        # Set early stopping
        early_stop_config = config.get("early_stopping", {})
        if early_stop_config:
            self.early_stopping_check.setChecked(True)
            self.monitor_metric_combo.setCurrentText(early_stop_config.get("monitor", "val_loss"))
            self.patience_spin.setValue(early_stop_config.get("patience", 10))
            self.min_delta_spin.setValue(early_stop_config.get("min_delta", 0.0001))
            self.mode_combo.setCurrentText(early_stop_config.get("mode", "min"))
        else:
            self.early_stopping_check.setChecked(False)

        # Set checkpointing
        checkpoint_config = config.get("checkpoint", {})
        save_top_k = checkpoint_config.get("save_top_k", 3)
        self.save_checkpoint_check.setChecked(save_top_k > 0)
        self.save_top_k_spin.setValue(save_top_k if save_top_k > 0 else 3)
        self.save_every_n_spin.setValue(checkpoint_config.get("save_every_n_epochs", 1))
        self.save_last_check.setChecked(checkpoint_config.get("save_last", True))

        # Update UI state
        self.on_optimizer_changed()
        self.on_scheduler_changed()

    def reset_configuration(self):
        """Reset to default configuration."""
        self.set_default_values()

    def set_default_values(self):
        """Set default values."""
        self.optimizer_combo.setCurrentText("AdamW")
        self.learning_rate_spin.setValue(0.001)
        self.weight_decay_spin.setValue(0.0001)
        self.momentum_spin.setValue(0.9)
        self.max_epochs_spin.setValue(100)
        self.gradient_clip_spin.setValue(1.0)
        self.accumulate_batches_spin.setValue(1)
        self.precision_combo.setCurrentText("16")

        self.scheduler_combo.setCurrentText("CosineAnnealing")
        self.step_size_spin.setValue(10)
        self.gamma_spin.setValue(0.1)

        self.early_stopping_check.setChecked(False)
        self.monitor_metric_combo.setCurrentText("val_loss")
        self.patience_spin.setValue(10)
        self.min_delta_spin.setValue(0.0001)
        self.mode_combo.setCurrentText("min")

        self.save_checkpoint_check.setChecked(True)
        self.save_top_k_spin.setValue(3)
        self.save_every_n_spin.setValue(1)
        self.save_last_check.setChecked(True)

        # Reset progress displays
        self.overall_progress.setValue(0)
        self.epoch_progress.setValue(0)
        self.current_epoch_label.setText("0 / 0")
        self.current_batch_label.setText("0 / 0")
        self.time_elapsed_label.setText("00:00:00")
        self.time_remaining_label.setText("00:00:00")
        self.train_loss_label.setText("0.0000")
        self.val_loss_label.setText("0.0000")
        self.train_acc_label.setText("0.00%")
        self.val_acc_label.setText("0.00%")
        self.lr_label.setText("0.0000")

        self.on_optimizer_changed()
        self.on_scheduler_changed()