"""Settings dialog for the application."""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QWidget, QLabel, QLineEdit, QSpinBox, QCheckBox,
    QPushButton, QComboBox, QGroupBox, QGridLayout,
    QFileDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QSettings


class SettingsDialog(QDialog):
    """Dialog for application settings."""

    def __init__(self, parent=None):
        """Initialize the settings dialog."""
        super().__init__(parent)
        self.settings = QSettings("VisionAI", "TrainingGUI")
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Settings")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        layout = QVBoxLayout()

        # Create tab widget
        self.tabs = QTabWidget()

        # General tab
        general_tab = self.create_general_tab()
        self.tabs.addTab(general_tab, "General")

        # Training tab
        training_tab = self.create_training_tab()
        self.tabs.addTab(training_tab, "Training")

        # Interface tab
        interface_tab = self.create_interface_tab()
        self.tabs.addTab(interface_tab, "Interface")

        # Advanced tab
        advanced_tab = self.create_advanced_tab()
        self.tabs.addTab(advanced_tab, "Advanced")

        layout.addWidget(self.tabs)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_settings)

        layout.addWidget(button_box)

        self.setLayout(layout)

    def create_general_tab(self) -> QWidget:
        """Create the general settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Paths group
        paths_group = QGroupBox("Paths")
        paths_layout = QGridLayout()

        paths_layout.addWidget(QLabel("Data Directory:"), 0, 0)
        self.data_dir_edit = QLineEdit()
        self.data_dir_browse = QPushButton("Browse...")
        self.data_dir_browse.clicked.connect(lambda: self.browse_directory(self.data_dir_edit))
        paths_layout.addWidget(self.data_dir_edit, 0, 1)
        paths_layout.addWidget(self.data_dir_browse, 0, 2)

        paths_layout.addWidget(QLabel("Output Directory:"), 1, 0)
        self.output_dir_edit = QLineEdit()
        self.output_dir_browse = QPushButton("Browse...")
        self.output_dir_browse.clicked.connect(lambda: self.browse_directory(self.output_dir_edit))
        paths_layout.addWidget(self.output_dir_edit, 1, 1)
        paths_layout.addWidget(self.output_dir_browse, 1, 2)

        paths_layout.addWidget(QLabel("Checkpoint Directory:"), 2, 0)
        self.checkpoint_dir_edit = QLineEdit()
        self.checkpoint_dir_browse = QPushButton("Browse...")
        self.checkpoint_dir_browse.clicked.connect(lambda: self.browse_directory(self.checkpoint_dir_edit))
        paths_layout.addWidget(self.checkpoint_dir_edit, 2, 1)
        paths_layout.addWidget(self.checkpoint_dir_browse, 2, 2)

        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)

        # Default values group
        defaults_group = QGroupBox("Default Values")
        defaults_layout = QGridLayout()

        defaults_layout.addWidget(QLabel("Default Batch Size:"), 0, 0)
        self.default_batch_size = QSpinBox()
        self.default_batch_size.setRange(1, 512)
        defaults_layout.addWidget(self.default_batch_size, 0, 1)

        defaults_layout.addWidget(QLabel("Default Epochs:"), 1, 0)
        self.default_epochs = QSpinBox()
        self.default_epochs.setRange(1, 1000)
        defaults_layout.addWidget(self.default_epochs, 1, 1)

        defaults_layout.addWidget(QLabel("Default Learning Rate:"), 2, 0)
        self.default_lr = QLineEdit()
        defaults_layout.addWidget(self.default_lr, 2, 1)

        defaults_group.setLayout(defaults_layout)
        layout.addWidget(defaults_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_training_tab(self) -> QWidget:
        """Create the training settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Hardware group
        hardware_group = QGroupBox("Hardware")
        hardware_layout = QGridLayout()

        hardware_layout.addWidget(QLabel("Default Device:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CPU", "GPU", "TPU"])
        hardware_layout.addWidget(self.device_combo, 0, 1)

        hardware_layout.addWidget(QLabel("Default Precision:"), 1, 0)
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["16", "32", "bf16"])
        hardware_layout.addWidget(self.precision_combo, 1, 1)

        hardware_layout.addWidget(QLabel("Num Workers:"), 2, 0)
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 16)
        hardware_layout.addWidget(self.num_workers_spin, 2, 1)

        self.pin_memory_check = QCheckBox("Pin Memory by Default")
        hardware_layout.addWidget(self.pin_memory_check, 3, 0, 1, 2)

        self.persistent_workers_check = QCheckBox("Persistent Workers by Default")
        hardware_layout.addWidget(self.persistent_workers_check, 4, 0, 1, 2)

        hardware_group.setLayout(hardware_layout)
        layout.addWidget(hardware_group)

        # Monitoring group
        monitoring_group = QGroupBox("Monitoring")
        monitoring_layout = QGridLayout()

        monitoring_layout.addWidget(QLabel("Log Every N Steps:"), 0, 0)
        self.log_every_n_spin = QSpinBox()
        self.log_every_n_spin.setRange(1, 1000)
        monitoring_layout.addWidget(self.log_every_n_spin, 0, 1)

        monitoring_layout.addWidget(QLabel("Val Check Interval:"), 1, 0)
        self.val_check_interval = QLineEdit()
        monitoring_layout.addWidget(self.val_check_interval, 1, 1)

        self.enable_profiler_check = QCheckBox("Enable Profiler")
        monitoring_layout.addWidget(self.enable_profiler_check, 2, 0, 1, 2)

        self.track_grad_norm_check = QCheckBox("Track Gradient Norm")
        monitoring_layout.addWidget(self.track_grad_norm_check, 3, 0, 1, 2)

        monitoring_group.setLayout(monitoring_layout)
        layout.addWidget(monitoring_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_interface_tab(self) -> QWidget:
        """Create the interface settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Appearance group
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QGridLayout()

        appearance_layout.addWidget(QLabel("Theme:"), 0, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Default", "Dark", "Light", "Auto"])
        appearance_layout.addWidget(self.theme_combo, 0, 1)

        appearance_layout.addWidget(QLabel("Font Size:"), 1, 0)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        appearance_layout.addWidget(self.font_size_spin, 1, 1)

        self.show_toolbar_check = QCheckBox("Show Toolbar")
        appearance_layout.addWidget(self.show_toolbar_check, 2, 0, 1, 2)

        self.show_statusbar_check = QCheckBox("Show Status Bar")
        appearance_layout.addWidget(self.show_statusbar_check, 3, 0, 1, 2)

        appearance_group.setLayout(appearance_layout)
        layout.addWidget(appearance_group)

        # Behavior group
        behavior_group = QGroupBox("Behavior")
        behavior_layout = QVBoxLayout()

        self.confirm_exit_check = QCheckBox("Confirm on Exit")
        behavior_layout.addWidget(self.confirm_exit_check)

        self.auto_save_check = QCheckBox("Auto-save Configuration")
        behavior_layout.addWidget(self.auto_save_check)

        self.restore_window_check = QCheckBox("Restore Window Position")
        behavior_layout.addWidget(self.restore_window_check)

        self.auto_scroll_logs_check = QCheckBox("Auto-scroll Logs")
        behavior_layout.addWidget(self.auto_scroll_logs_check)

        behavior_group.setLayout(behavior_layout)
        layout.addWidget(behavior_group)

        # Plots group
        plots_group = QGroupBox("Plots")
        plots_layout = QGridLayout()

        plots_layout.addWidget(QLabel("Update Interval (ms):"), 0, 0)
        self.plot_update_interval = QSpinBox()
        self.plot_update_interval.setRange(100, 10000)
        plots_layout.addWidget(self.plot_update_interval, 0, 1)

        plots_layout.addWidget(QLabel("Max Points:"), 1, 0)
        self.max_plot_points = QSpinBox()
        self.max_plot_points.setRange(100, 10000)
        plots_layout.addWidget(self.max_plot_points, 1, 1)

        self.smooth_plots_check = QCheckBox("Smooth Plots by Default")
        plots_layout.addWidget(self.smooth_plots_check, 2, 0, 1, 2)

        plots_group.setLayout(plots_layout)
        layout.addWidget(plots_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_advanced_tab(self) -> QWidget:
        """Create the advanced settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Logging group
        logging_group = QGroupBox("Logging")
        logging_layout = QGridLayout()

        logging_layout.addWidget(QLabel("Log Level:"), 0, 0)
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        logging_layout.addWidget(self.log_level_combo, 0, 1)

        logging_layout.addWidget(QLabel("Max Log Lines:"), 1, 0)
        self.max_log_lines = QSpinBox()
        self.max_log_lines.setRange(100, 100000)
        logging_layout.addWidget(self.max_log_lines, 1, 1)

        self.save_logs_check = QCheckBox("Save Logs to File")
        logging_layout.addWidget(self.save_logs_check, 2, 0, 1, 2)

        logging_group.setLayout(logging_layout)
        layout.addWidget(logging_group)

        # Experiment tracking group
        tracking_group = QGroupBox("Experiment Tracking")
        tracking_layout = QGridLayout()

        tracking_layout.addWidget(QLabel("MLflow URI:"), 0, 0)
        self.mlflow_uri_edit = QLineEdit()
        tracking_layout.addWidget(self.mlflow_uri_edit, 0, 1)

        self.enable_mlflow_check = QCheckBox("Enable MLflow Tracking")
        tracking_layout.addWidget(self.enable_mlflow_check, 1, 0, 1, 2)

        self.enable_tensorboard_check = QCheckBox("Enable TensorBoard")
        tracking_layout.addWidget(self.enable_tensorboard_check, 2, 0, 1, 2)

        tracking_group.setLayout(tracking_layout)
        layout.addWidget(tracking_group)

        # Cache group
        cache_group = QGroupBox("Cache")
        cache_layout = QVBoxLayout()

        self.clear_cache_button = QPushButton("Clear Cache")
        cache_layout.addWidget(self.clear_cache_button)

        self.clear_logs_button = QPushButton("Clear Old Logs")
        cache_layout.addWidget(self.clear_logs_button)

        self.clear_checkpoints_button = QPushButton("Clear Old Checkpoints")
        cache_layout.addWidget(self.clear_checkpoints_button)

        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def browse_directory(self, line_edit: QLineEdit):
        """Browse for a directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            line_edit.text()
        )
        if directory:
            line_edit.setText(directory)

    def load_settings(self):
        """Load settings from QSettings."""
        # General settings
        self.data_dir_edit.setText(self.settings.value("paths/data_dir", "./data"))
        self.output_dir_edit.setText(self.settings.value("paths/output_dir", "./outputs"))
        self.checkpoint_dir_edit.setText(self.settings.value("paths/checkpoint_dir", "./checkpoints"))
        self.default_batch_size.setValue(int(self.settings.value("defaults/batch_size", 32)))
        self.default_epochs.setValue(int(self.settings.value("defaults/epochs", 100)))
        self.default_lr.setText(self.settings.value("defaults/learning_rate", "0.001"))

        # Training settings
        self.device_combo.setCurrentText(self.settings.value("hardware/device", "Auto"))
        self.precision_combo.setCurrentText(self.settings.value("hardware/precision", "16"))
        self.num_workers_spin.setValue(int(self.settings.value("hardware/num_workers", 4)))
        self.pin_memory_check.setChecked(self.settings.value("hardware/pin_memory", True, type=bool))
        self.persistent_workers_check.setChecked(self.settings.value("hardware/persistent_workers", True, type=bool))
        self.log_every_n_spin.setValue(int(self.settings.value("monitoring/log_every_n", 10)))
        self.val_check_interval.setText(self.settings.value("monitoring/val_check_interval", "1.0"))
        self.enable_profiler_check.setChecked(self.settings.value("monitoring/enable_profiler", False, type=bool))
        self.track_grad_norm_check.setChecked(self.settings.value("monitoring/track_grad_norm", False, type=bool))

        # Interface settings
        self.theme_combo.setCurrentText(self.settings.value("appearance/theme", "Default"))
        self.font_size_spin.setValue(int(self.settings.value("appearance/font_size", 10)))
        self.show_toolbar_check.setChecked(self.settings.value("appearance/show_toolbar", True, type=bool))
        self.show_statusbar_check.setChecked(self.settings.value("appearance/show_statusbar", True, type=bool))
        self.confirm_exit_check.setChecked(self.settings.value("behavior/confirm_exit", True, type=bool))
        self.auto_save_check.setChecked(self.settings.value("behavior/auto_save", False, type=bool))
        self.restore_window_check.setChecked(self.settings.value("behavior/restore_window", True, type=bool))
        self.auto_scroll_logs_check.setChecked(self.settings.value("behavior/auto_scroll_logs", True, type=bool))
        self.plot_update_interval.setValue(int(self.settings.value("plots/update_interval", 1000)))
        self.max_plot_points.setValue(int(self.settings.value("plots/max_points", 1000)))
        self.smooth_plots_check.setChecked(self.settings.value("plots/smooth_by_default", False, type=bool))

        # Advanced settings
        self.log_level_combo.setCurrentText(self.settings.value("logging/level", "INFO"))
        self.max_log_lines.setValue(int(self.settings.value("logging/max_lines", 10000)))
        self.save_logs_check.setChecked(self.settings.value("logging/save_to_file", False, type=bool))
        self.mlflow_uri_edit.setText(self.settings.value("tracking/mlflow_uri", ""))
        self.enable_mlflow_check.setChecked(self.settings.value("tracking/enable_mlflow", False, type=bool))
        self.enable_tensorboard_check.setChecked(self.settings.value("tracking/enable_tensorboard", False, type=bool))

    def save_settings(self):
        """Save settings to QSettings."""
        # General settings
        self.settings.setValue("paths/data_dir", self.data_dir_edit.text())
        self.settings.setValue("paths/output_dir", self.output_dir_edit.text())
        self.settings.setValue("paths/checkpoint_dir", self.checkpoint_dir_edit.text())
        self.settings.setValue("defaults/batch_size", self.default_batch_size.value())
        self.settings.setValue("defaults/epochs", self.default_epochs.value())
        self.settings.setValue("defaults/learning_rate", self.default_lr.text())

        # Training settings
        self.settings.setValue("hardware/device", self.device_combo.currentText())
        self.settings.setValue("hardware/precision", self.precision_combo.currentText())
        self.settings.setValue("hardware/num_workers", self.num_workers_spin.value())
        self.settings.setValue("hardware/pin_memory", self.pin_memory_check.isChecked())
        self.settings.setValue("hardware/persistent_workers", self.persistent_workers_check.isChecked())
        self.settings.setValue("monitoring/log_every_n", self.log_every_n_spin.value())
        self.settings.setValue("monitoring/val_check_interval", self.val_check_interval.text())
        self.settings.setValue("monitoring/enable_profiler", self.enable_profiler_check.isChecked())
        self.settings.setValue("monitoring/track_grad_norm", self.track_grad_norm_check.isChecked())

        # Interface settings
        self.settings.setValue("appearance/theme", self.theme_combo.currentText())
        self.settings.setValue("appearance/font_size", self.font_size_spin.value())
        self.settings.setValue("appearance/show_toolbar", self.show_toolbar_check.isChecked())
        self.settings.setValue("appearance/show_statusbar", self.show_statusbar_check.isChecked())
        self.settings.setValue("behavior/confirm_exit", self.confirm_exit_check.isChecked())
        self.settings.setValue("behavior/auto_save", self.auto_save_check.isChecked())
        self.settings.setValue("behavior/restore_window", self.restore_window_check.isChecked())
        self.settings.setValue("behavior/auto_scroll_logs", self.auto_scroll_logs_check.isChecked())
        self.settings.setValue("plots/update_interval", self.plot_update_interval.value())
        self.settings.setValue("plots/max_points", self.max_plot_points.value())
        self.settings.setValue("plots/smooth_by_default", self.smooth_plots_check.isChecked())

        # Advanced settings
        self.settings.setValue("logging/level", self.log_level_combo.currentText())
        self.settings.setValue("logging/max_lines", self.max_log_lines.value())
        self.settings.setValue("logging/save_to_file", self.save_logs_check.isChecked())
        self.settings.setValue("tracking/mlflow_uri", self.mlflow_uri_edit.text())
        self.settings.setValue("tracking/enable_mlflow", self.enable_mlflow_check.isChecked())
        self.settings.setValue("tracking/enable_tensorboard", self.enable_tensorboard_check.isChecked())

    def apply_settings(self):
        """Apply settings without closing dialog."""
        self.save_settings()

    def accept(self):
        """Accept and save settings."""
        self.save_settings()
        super().accept()