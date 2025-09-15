"""Main window for Vision AI Training GUI."""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMenuBar, QMenu, QAction, QStatusBar,
    QMessageBox, QFileDialog, QTabWidget, QDockWidget,
    QToolBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSettings
from PyQt5.QtGui import QIcon, QKeySequence

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui.widgets.model_config_widget import ModelConfigWidget
from gui.widgets.dataset_config_widget import DatasetConfigWidget
from gui.widgets.training_control_widget import TrainingControlWidget
from gui.widgets.metrics_visualization_widget import MetricsVisualizationWidget
from gui.widgets.log_viewer_widget import LogViewerWidget
from gui.widgets.experiment_tracker_widget import ExperimentTrackerWidget
from gui.threads.training_thread import TrainingThread
from gui.utils.config_manager import ConfigManager
from gui.dialogs.settings_dialog import SettingsDialog
from gui.dialogs.about_dialog import AboutDialog


class VisionTrainingMainWindow(QMainWindow):
    """Main window for Vision AI training GUI application."""

    # Signals
    training_started = pyqtSignal()
    training_stopped = pyqtSignal()
    training_completed = pyqtSignal()
    config_changed = pyqtSignal(dict)

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        # Initialize components
        self.training_thread = None
        self.config_manager = ConfigManager()
        self.settings = QSettings("VisionAI", "TrainingGUI")
        self.current_config = {}

        # Setup UI
        self.init_ui()
        self.setup_connections()
        self.restore_state()

        # Status timer for updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Vision AI Training Studio")
        self.setGeometry(100, 100, 1400, 900)

        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #007ACC;
            }
        """)

        # Create menu bar
        self.create_menu_bar()

        # Create toolbar
        self.create_toolbar()

        # Create central widget
        self.create_central_widget()

        # Create dock widgets
        self.create_dock_widgets()

        # Create status bar
        self.create_status_bar()

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Configuration", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.new_configuration)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Configuration...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_configuration)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Configuration", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_configuration)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save Configuration &As...", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self.save_configuration_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        recent_menu = file_menu.addMenu("Recent Configurations")
        self.update_recent_menu(recent_menu)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut(QKeySequence.Preferences)
        settings_action.triggered.connect(self.show_settings)
        edit_menu.addAction(settings_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        self.log_dock_action = QAction("&Log Viewer", self, checkable=True, checked=True)
        self.log_dock_action.triggered.connect(self.toggle_log_viewer)
        view_menu.addAction(self.log_dock_action)

        self.experiment_dock_action = QAction("&Experiment Tracker", self, checkable=True, checked=True)
        self.experiment_dock_action.triggered.connect(self.toggle_experiment_tracker)
        view_menu.addAction(self.experiment_dock_action)

        view_menu.addSeparator()

        reset_layout_action = QAction("&Reset Layout", self)
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)

        # Training menu
        training_menu = menubar.addMenu("&Training")

        self.start_training_action = QAction("&Start Training", self)
        self.start_training_action.setShortcut(QKeySequence("F5"))
        self.start_training_action.triggered.connect(self.start_training)
        training_menu.addAction(self.start_training_action)

        self.stop_training_action = QAction("&Stop Training", self)
        self.stop_training_action.setShortcut(QKeySequence("Shift+F5"))
        self.stop_training_action.triggered.connect(self.stop_training)
        self.stop_training_action.setEnabled(False)
        training_menu.addAction(self.stop_training_action)

        training_menu.addSeparator()

        validate_config_action = QAction("&Validate Configuration", self)
        validate_config_action.triggered.connect(self.validate_configuration)
        training_menu.addAction(validate_config_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        documentation_action = QAction("&Documentation", self)
        documentation_action.setShortcut(QKeySequence.HelpContents)
        documentation_action.triggered.connect(self.show_documentation)
        help_menu.addAction(documentation_action)

        help_menu.addSeparator()

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        """Create the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Add actions to toolbar
        toolbar.addAction("New", self.new_configuration)
        toolbar.addAction("Open", self.open_configuration)
        toolbar.addAction("Save", self.save_configuration)
        toolbar.addSeparator()

        self.start_button_action = toolbar.addAction("Start Training", self.start_training)
        self.stop_button_action = toolbar.addAction("Stop Training", self.stop_training)
        self.stop_button_action.setEnabled(False)

        toolbar.addSeparator()
        toolbar.addAction("Settings", self.show_settings)

    def create_central_widget(self):
        """Create the central widget with tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Configuration tab
        config_tab = QWidget()
        config_layout = QHBoxLayout(config_tab)

        # Left side - Model and Dataset configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.model_config_widget = ModelConfigWidget()
        self.dataset_config_widget = DatasetConfigWidget()

        left_layout.addWidget(self.model_config_widget)
        left_layout.addWidget(self.dataset_config_widget)
        left_layout.addStretch()

        # Right side - Training control
        self.training_control_widget = TrainingControlWidget()

        # Add to splitter
        config_splitter = QSplitter(Qt.Horizontal)
        config_splitter.addWidget(left_panel)
        config_splitter.addWidget(self.training_control_widget)
        config_splitter.setStretchFactor(0, 1)
        config_splitter.setStretchFactor(1, 1)

        config_layout.addWidget(config_splitter)

        # Metrics tab
        self.metrics_widget = MetricsVisualizationWidget()

        # Add tabs
        self.tab_widget.addTab(config_tab, "Configuration")
        self.tab_widget.addTab(self.metrics_widget, "Metrics & Visualization")

        layout.addWidget(self.tab_widget)

    def create_dock_widgets(self):
        """Create dockable widgets."""
        # Log viewer dock
        self.log_dock = QDockWidget("Log Viewer", self)
        self.log_viewer = LogViewerWidget()
        self.log_dock.setWidget(self.log_viewer)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)

        # Experiment tracker dock
        self.experiment_dock = QDockWidget("Experiment Tracker", self)
        self.experiment_tracker = ExperimentTrackerWidget()
        self.experiment_dock.setWidget(self.experiment_tracker)
        self.addDockWidget(Qt.RightDockWidgetArea, self.experiment_dock)

    def create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add permanent widgets
        self.status_label = QWidget()
        status_layout = QHBoxLayout(self.status_label)
        status_layout.setContentsMargins(0, 0, 0, 0)

        from PyQt5.QtWidgets import QLabel
        self.training_status = QLabel("Ready")
        self.gpu_status = QLabel("GPU: N/A")
        self.memory_status = QLabel("Memory: N/A")

        status_layout.addWidget(self.training_status)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.gpu_status)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.memory_status)

        self.status_bar.addPermanentWidget(self.status_label)

    def setup_connections(self):
        """Setup signal-slot connections."""
        # Model configuration changes
        self.model_config_widget.config_changed.connect(self.on_model_config_changed)

        # Dataset configuration changes
        self.dataset_config_widget.config_changed.connect(self.on_dataset_config_changed)

        # Training control signals
        self.training_control_widget.start_training.connect(self.start_training)
        self.training_control_widget.stop_training.connect(self.stop_training)

        # Training thread signals (will be connected when thread is created)

    def on_model_config_changed(self, config: Dict[str, Any]):
        """Handle model configuration changes."""
        self.current_config["model"] = config
        self.config_changed.emit(self.current_config)

    def on_dataset_config_changed(self, config: Dict[str, Any]):
        """Handle dataset configuration changes."""
        self.current_config["data"] = config
        self.config_changed.emit(self.current_config)

    def start_training(self):
        """Start the training process."""
        # Validate configuration
        if not self.validate_configuration(show_message=True):
            return

        # Collect all configurations
        config = self.collect_configuration()

        # Create and start training thread
        self.training_thread = TrainingThread(config)

        # Connect signals
        self.training_thread.training_started.connect(self.on_training_started)
        self.training_thread.training_progress.connect(self.on_training_progress)
        self.training_thread.training_completed.connect(self.on_training_completed)
        self.training_thread.training_error.connect(self.on_training_error)
        self.training_thread.log_message.connect(self.log_viewer.add_log)
        self.training_thread.metrics_update.connect(self.metrics_widget.update_metrics)

        # Start training
        self.training_thread.start()

        # Update UI
        self.start_training_action.setEnabled(False)
        self.start_button_action.setEnabled(False)
        self.stop_training_action.setEnabled(True)
        self.stop_button_action.setEnabled(True)

        # Switch to metrics tab
        self.tab_widget.setCurrentIndex(1)

        # Emit signal
        self.training_started.emit()

    def stop_training(self):
        """Stop the training process."""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Stop Training",
                "Are you sure you want to stop the training?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.training_thread.stop()
                self.training_thread.wait()

                # Update UI
                self.start_training_action.setEnabled(True)
                self.start_button_action.setEnabled(True)
                self.stop_training_action.setEnabled(False)
                self.stop_button_action.setEnabled(False)

                # Emit signal
                self.training_stopped.emit()

                self.status_bar.showMessage("Training stopped", 3000)

    def on_training_started(self):
        """Handle training started event."""
        self.training_control_widget.set_training_active(True)
        self.log_viewer.add_log("Training started", "INFO")
        self.status_bar.showMessage("Training in progress...")

    def on_training_progress(self, progress: Dict[str, Any]):
        """Handle training progress update."""
        self.training_control_widget.update_progress(progress)

        # Update experiment tracker
        self.experiment_tracker.update_current_experiment(progress)

    def on_training_completed(self):
        """Handle training completed event."""
        self.training_control_widget.set_training_active(False)
        self.log_viewer.add_log("Training completed successfully", "SUCCESS")

        # Update UI
        self.start_training_action.setEnabled(True)
        self.start_button_action.setEnabled(True)
        self.stop_training_action.setEnabled(False)
        self.stop_button_action.setEnabled(False)

        # Emit signal
        self.training_completed.emit()

        self.status_bar.showMessage("Training completed", 3000)

        # Show completion message
        QMessageBox.information(
            self,
            "Training Complete",
            "Training has completed successfully!"
        )

    def on_training_error(self, error_message: str):
        """Handle training error."""
        self.training_control_widget.set_training_active(False)
        self.log_viewer.add_log(f"Training error: {error_message}", "ERROR")

        # Update UI
        self.start_training_action.setEnabled(True)
        self.start_button_action.setEnabled(True)
        self.stop_training_action.setEnabled(False)
        self.stop_button_action.setEnabled(False)

        self.status_bar.showMessage("Training failed", 3000)

        # Show error message
        QMessageBox.critical(
            self,
            "Training Error",
            f"An error occurred during training:\n{error_message}"
        )

    def collect_configuration(self) -> Dict[str, Any]:
        """Collect configuration from all widgets."""
        config = {
            "model": self.model_config_widget.get_configuration(),
            "data": self.dataset_config_widget.get_configuration(),
            "training": self.training_control_widget.get_configuration(),
            "experiment": {
                "name": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "seed": 42
            }
        }
        return config

    def validate_configuration(self, show_message: bool = False) -> bool:
        """Validate the current configuration."""
        errors = []

        # Validate model configuration
        model_config = self.model_config_widget.get_configuration()
        if not model_config.get("name"):
            errors.append("Model type not selected")

        # Validate dataset configuration
        dataset_config = self.dataset_config_widget.get_configuration()
        if not dataset_config.get("name"):
            errors.append("Dataset not selected")

        # Validate training configuration
        training_config = self.training_control_widget.get_configuration()
        if training_config.get("max_epochs", 0) <= 0:
            errors.append("Number of epochs must be greater than 0")

        if errors and show_message:
            QMessageBox.warning(
                self,
                "Configuration Validation",
                "Please fix the following issues:\n\n" + "\n".join(f"• {e}" for e in errors)
            )

        return len(errors) == 0

    def new_configuration(self):
        """Create a new configuration."""
        reply = QMessageBox.question(
            self,
            "New Configuration",
            "Create a new configuration? Any unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.model_config_widget.reset_configuration()
            self.dataset_config_widget.reset_configuration()
            self.training_control_widget.reset_configuration()
            self.current_config = {}
            self.status_bar.showMessage("New configuration created", 3000)

    def open_configuration(self):
        """Open a configuration file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Configuration",
            str(Path.home()),
            "JSON Files (*.json);;YAML Files (*.yaml *.yml);;All Files (*.*)"
        )

        if filename:
            try:
                config = self.config_manager.load_config(filename)
                self.apply_configuration(config)
                self.add_to_recent(filename)
                self.status_bar.showMessage(f"Configuration loaded from {filename}", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Configuration",
                    f"Failed to load configuration:\n{str(e)}"
                )

    def save_configuration(self):
        """Save the current configuration."""
        if not hasattr(self, "current_config_file") or not self.current_config_file:
            self.save_configuration_as()
        else:
            self.save_configuration_to_file(self.current_config_file)

    def save_configuration_as(self):
        """Save the configuration to a new file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            str(Path.home()),
            "JSON Files (*.json);;YAML Files (*.yaml);;All Files (*.*)"
        )

        if filename:
            self.save_configuration_to_file(filename)
            self.current_config_file = filename
            self.add_to_recent(filename)

    def save_configuration_to_file(self, filename: str):
        """Save configuration to a specific file."""
        try:
            config = self.collect_configuration()
            self.config_manager.save_config(config, filename)
            self.status_bar.showMessage(f"Configuration saved to {filename}", 3000)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Configuration",
                f"Failed to save configuration:\n{str(e)}"
            )

    def apply_configuration(self, config: Dict[str, Any]):
        """Apply a configuration to all widgets."""
        if "model" in config:
            self.model_config_widget.set_configuration(config["model"])
        if "data" in config:
            self.dataset_config_widget.set_configuration(config["data"])
        if "training" in config:
            self.training_control_widget.set_configuration(config["training"])

        self.current_config = config

    def add_to_recent(self, filename: str):
        """Add a file to the recent files list."""
        recent_files = self.settings.value("recent_files", [])
        if filename in recent_files:
            recent_files.remove(filename)
        recent_files.insert(0, filename)
        recent_files = recent_files[:10]  # Keep only 10 recent files
        self.settings.setValue("recent_files", recent_files)

    def update_recent_menu(self, menu: QMenu):
        """Update the recent files menu."""
        menu.clear()
        recent_files = self.settings.value("recent_files", [])

        for filename in recent_files:
            if Path(filename).exists():
                action = QAction(Path(filename).name, self)
                action.setData(filename)
                action.triggered.connect(lambda checked, f=filename: self.open_recent_file(f))
                menu.addAction(action)

        if menu.isEmpty():
            action = QAction("No recent files", self)
            action.setEnabled(False)
            menu.addAction(action)

    def open_recent_file(self, filename: str):
        """Open a recent configuration file."""
        if Path(filename).exists():
            try:
                config = self.config_manager.load_config(filename)
                self.apply_configuration(config)
                self.current_config_file = filename
                self.status_bar.showMessage(f"Configuration loaded from {filename}", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Configuration",
                    f"Failed to load configuration:\n{str(e)}"
                )

    def show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec_():
            # Apply settings
            self.apply_settings()

    def apply_settings(self):
        """Apply application settings."""
        # Reload settings and apply them
        pass

    def toggle_log_viewer(self):
        """Toggle the log viewer dock."""
        if self.log_dock.isVisible():
            self.log_dock.hide()
        else:
            self.log_dock.show()

    def toggle_experiment_tracker(self):
        """Toggle the experiment tracker dock."""
        if self.experiment_dock.isVisible():
            self.experiment_dock.hide()
        else:
            self.experiment_dock.show()

    def reset_layout(self):
        """Reset the window layout to default."""
        self.log_dock.show()
        self.experiment_dock.show()
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.experiment_dock)

    def show_documentation(self):
        """Show the documentation."""
        QMessageBox.information(
            self,
            "Documentation",
            "Documentation will be available at:\nhttps://github.com/your-repo/vision-ai-training"
        )

    def show_about(self):
        """Show the about dialog."""
        dialog = AboutDialog(self)
        dialog.exec_()

    def update_status(self):
        """Update the status bar."""
        if self.training_thread and self.training_thread.isRunning():
            self.training_status.setText("Training Active")
        else:
            self.training_status.setText("Ready")

        # Update GPU and memory status
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_status.setText(f"GPU: {torch.cuda.get_device_name(0)}")
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.memory_status.setText(f"Memory: {memory_used:.1f}/{memory_total:.1f} GB")
            else:
                self.gpu_status.setText("GPU: CPU Mode")
                import psutil
                memory_percent = psutil.virtual_memory().percent
                self.memory_status.setText(f"Memory: {memory_percent:.1f}%")
        except:
            pass

    def restore_state(self):
        """Restore the window state from settings."""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)

    def closeEvent(self, event):
        """Handle the window close event."""
        # Check if training is running
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Training in Progress",
                "Training is still running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                event.ignore()
                return
            else:
                self.training_thread.stop()
                self.training_thread.wait()

        # Save window state
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

        event.accept()


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Vision AI Training Studio")
    app.setOrganizationName("VisionAI")

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = VisionTrainingMainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    main()