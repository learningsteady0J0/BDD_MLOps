"""Widget for tracking experiments."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget,
    QTreeWidgetItem, QPushButton, QLabel, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMenu, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QColor, QBrush, QIcon


class ExperimentTrackerWidget(QWidget):
    """Widget for tracking and managing experiments."""

    # Signals
    experiment_selected = pyqtSignal(dict)
    compare_experiments = pyqtSignal(list)

    def __init__(self, parent=None):
        """Initialize the experiment tracker widget."""
        super().__init__(parent)

        self.experiments = []
        self.current_experiment = None

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Current experiment info
        current_group = QGroupBox("Current Experiment")
        current_layout = QVBoxLayout()

        # Experiment details
        details_layout = QGridLayout()
        from PyQt5.QtWidgets import QGridLayout

        details_layout.addWidget(QLabel("Name:"), 0, 0)
        self.exp_name_label = QLabel("N/A")
        self.exp_name_label.setStyleSheet("font-weight: bold;")
        details_layout.addWidget(self.exp_name_label, 0, 1)

        details_layout.addWidget(QLabel("Status:"), 1, 0)
        self.exp_status_label = QLabel("Not Started")
        details_layout.addWidget(self.exp_status_label, 1, 1)

        details_layout.addWidget(QLabel("Start Time:"), 2, 0)
        self.exp_start_time_label = QLabel("N/A")
        details_layout.addWidget(self.exp_start_time_label, 2, 1)

        details_layout.addWidget(QLabel("Duration:"), 3, 0)
        self.exp_duration_label = QLabel("00:00:00")
        details_layout.addWidget(self.exp_duration_label, 3, 1)

        details_layout.addWidget(QLabel("Best Val Loss:"), 4, 0)
        self.exp_best_loss_label = QLabel("N/A")
        self.exp_best_loss_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        details_layout.addWidget(self.exp_best_loss_label, 4, 1)

        details_layout.addWidget(QLabel("Best Val Acc:"), 5, 0)
        self.exp_best_acc_label = QLabel("N/A")
        self.exp_best_acc_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        details_layout.addWidget(self.exp_best_acc_label, 5, 1)

        current_layout.addLayout(details_layout)

        # Current experiment actions
        current_actions_layout = QHBoxLayout()
        self.save_exp_button = QPushButton("Save")
        self.tag_exp_button = QPushButton("Add Tag")
        self.notes_exp_button = QPushButton("Add Notes")
        current_actions_layout.addWidget(self.save_exp_button)
        current_actions_layout.addWidget(self.tag_exp_button)
        current_actions_layout.addWidget(self.notes_exp_button)
        current_layout.addLayout(current_actions_layout)

        current_group.setLayout(current_layout)
        layout.addWidget(current_group)

        # Experiment history
        history_group = QGroupBox("Experiment History")
        history_layout = QVBoxLayout()

        # Controls
        history_controls_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.compare_button = QPushButton("Compare Selected")
        self.delete_button = QPushButton("Delete Selected")
        history_controls_layout.addWidget(self.refresh_button)
        history_controls_layout.addWidget(self.compare_button)
        history_controls_layout.addWidget(self.delete_button)
        history_controls_layout.addStretch()
        history_layout.addLayout(history_controls_layout)

        # Experiment tree
        self.experiment_tree = QTreeWidget()
        self.experiment_tree.setHeaderLabels([
            "Name", "Model", "Dataset", "Val Loss", "Val Acc", "Status", "Date"
        ])
        self.experiment_tree.setAlternatingRowColors(True)
        self.experiment_tree.setSortingEnabled(True)
        self.experiment_tree.setSelectionMode(QTreeWidget.ExtendedSelection)

        # Set column widths
        self.experiment_tree.setColumnWidth(0, 150)
        self.experiment_tree.setColumnWidth(1, 80)
        self.experiment_tree.setColumnWidth(2, 80)
        self.experiment_tree.setColumnWidth(3, 80)
        self.experiment_tree.setColumnWidth(4, 80)
        self.experiment_tree.setColumnWidth(5, 80)

        # Context menu
        self.experiment_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.experiment_tree.customContextMenuRequested.connect(self.show_context_menu)

        history_layout.addWidget(self.experiment_tree)

        # Statistics
        stats_layout = QHBoxLayout()
        self.total_exp_label = QLabel("Total Experiments: 0")
        self.success_exp_label = QLabel("Successful: 0")
        self.failed_exp_label = QLabel("Failed: 0")
        stats_layout.addWidget(self.total_exp_label)
        stats_layout.addWidget(self.success_exp_label)
        stats_layout.addWidget(self.failed_exp_label)
        stats_layout.addStretch()
        history_layout.addLayout(stats_layout)

        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

        self.setLayout(layout)

        # Timer for updating current experiment
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_current_duration)
        self.update_timer.start(1000)  # Update every second

    def setup_connections(self):
        """Setup signal-slot connections."""
        self.refresh_button.clicked.connect(self.refresh_experiments)
        self.compare_button.clicked.connect(self.compare_selected)
        self.delete_button.clicked.connect(self.delete_selected)
        self.save_exp_button.clicked.connect(self.save_current_experiment)
        self.tag_exp_button.clicked.connect(self.add_tag_to_current)
        self.notes_exp_button.clicked.connect(self.add_notes_to_current)
        self.experiment_tree.itemDoubleClicked.connect(self.on_experiment_double_clicked)

    def start_new_experiment(self, config: Dict[str, Any]):
        """Start tracking a new experiment."""
        self.current_experiment = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "name": config.get("experiment", {}).get("name", "Unnamed"),
            "model": config.get("model", {}).get("name", "Unknown"),
            "dataset": config.get("data", {}).get("name", "Unknown"),
            "config": config,
            "start_time": datetime.now(),
            "status": "Running",
            "metrics": {
                "train_loss": [],
                "val_loss": [],
                "train_acc": [],
                "val_acc": []
            },
            "best_val_loss": float('inf'),
            "best_val_acc": 0.0,
            "tags": [],
            "notes": ""
        }

        self.update_current_experiment_display()
        self.add_log(f"Started experiment: {self.current_experiment['name']}", "INFO")

    def update_current_experiment(self, metrics: Dict[str, Any]):
        """Update current experiment with new metrics."""
        if not self.current_experiment:
            return

        # Update metrics history
        for key in ["train_loss", "val_loss", "train_acc", "val_acc"]:
            if key in metrics:
                self.current_experiment["metrics"][key].append(metrics[key])

        # Update best metrics
        if "val_loss" in metrics:
            if metrics["val_loss"] < self.current_experiment["best_val_loss"]:
                self.current_experiment["best_val_loss"] = metrics["val_loss"]
                self.exp_best_loss_label.setText(f"{metrics['val_loss']:.4f}")

        if "val_acc" in metrics:
            if metrics["val_acc"] > self.current_experiment["best_val_acc"]:
                self.current_experiment["best_val_acc"] = metrics["val_acc"]
                self.exp_best_acc_label.setText(f"{metrics['val_acc']:.2f}%")

        # Update status if training completed
        if metrics.get("training_completed", False):
            self.current_experiment["status"] = "Completed"
            self.current_experiment["end_time"] = datetime.now()
            self.exp_status_label.setText("Completed")
            self.exp_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

    def update_current_experiment_display(self):
        """Update the current experiment display."""
        if not self.current_experiment:
            return

        self.exp_name_label.setText(self.current_experiment["name"])
        self.exp_status_label.setText(self.current_experiment["status"])

        # Set status color
        if self.current_experiment["status"] == "Running":
            self.exp_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        elif self.current_experiment["status"] == "Completed":
            self.exp_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif self.current_experiment["status"] == "Failed":
            self.exp_status_label.setStyleSheet("color: #F44336; font-weight: bold;")

        start_time = self.current_experiment["start_time"]
        self.exp_start_time_label.setText(start_time.strftime("%H:%M:%S"))

        if self.current_experiment["best_val_loss"] != float('inf'):
            self.exp_best_loss_label.setText(f"{self.current_experiment['best_val_loss']:.4f}")
        if self.current_experiment["best_val_acc"] > 0:
            self.exp_best_acc_label.setText(f"{self.current_experiment['best_val_acc']:.2f}%")

    def update_current_duration(self):
        """Update the duration of the current experiment."""
        if self.current_experiment and self.current_experiment["status"] == "Running":
            duration = datetime.now() - self.current_experiment["start_time"]
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.exp_duration_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def complete_current_experiment(self, success: bool = True):
        """Mark current experiment as completed."""
        if not self.current_experiment:
            return

        self.current_experiment["status"] = "Completed" if success else "Failed"
        self.current_experiment["end_time"] = datetime.now()

        # Add to experiment history
        self.experiments.append(self.current_experiment.copy())
        self.add_experiment_to_tree(self.current_experiment)

        # Update statistics
        self.update_statistics()

        # Update display
        self.update_current_experiment_display()

    def add_experiment_to_tree(self, experiment: Dict[str, Any]):
        """Add an experiment to the tree widget."""
        item = QTreeWidgetItem()

        item.setText(0, experiment["name"])
        item.setText(1, experiment["model"])
        item.setText(2, experiment["dataset"])

        if experiment["best_val_loss"] != float('inf'):
            item.setText(3, f"{experiment['best_val_loss']:.4f}")
        else:
            item.setText(3, "N/A")

        if experiment["best_val_acc"] > 0:
            item.setText(4, f"{experiment['best_val_acc']:.2f}%")
        else:
            item.setText(4, "N/A")

        item.setText(5, experiment["status"])
        item.setText(6, experiment["start_time"].strftime("%Y-%m-%d %H:%M"))

        # Set item data
        item.setData(0, Qt.UserRole, experiment)

        # Color based on status
        if experiment["status"] == "Completed":
            for i in range(7):
                item.setForeground(i, QBrush(QColor("#4CAF50")))
        elif experiment["status"] == "Failed":
            for i in range(7):
                item.setForeground(i, QBrush(QColor("#F44336")))
        elif experiment["status"] == "Running":
            for i in range(7):
                item.setForeground(i, QBrush(QColor("#2196F3")))

        self.experiment_tree.addTopLevelItem(item)

    def refresh_experiments(self):
        """Refresh the experiment list."""
        # Clear tree
        self.experiment_tree.clear()

        # Re-add all experiments
        for exp in self.experiments:
            self.add_experiment_to_tree(exp)

        # Update statistics
        self.update_statistics()

    def update_statistics(self):
        """Update experiment statistics."""
        total = len(self.experiments)
        successful = sum(1 for exp in self.experiments if exp["status"] == "Completed")
        failed = sum(1 for exp in self.experiments if exp["status"] == "Failed")

        self.total_exp_label.setText(f"Total Experiments: {total}")
        self.success_exp_label.setText(f"Successful: {successful}")
        self.failed_exp_label.setText(f"Failed: {failed}")

    def compare_selected(self):
        """Compare selected experiments."""
        selected_items = self.experiment_tree.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(
                self,
                "Selection Error",
                "Please select at least 2 experiments to compare."
            )
            return

        experiments = [item.data(0, Qt.UserRole) for item in selected_items]
        self.compare_experiments.emit(experiments)

    def delete_selected(self):
        """Delete selected experiments."""
        selected_items = self.experiment_tree.selectedItems()
        if not selected_items:
            return

        reply = QMessageBox.question(
            self,
            "Delete Experiments",
            f"Are you sure you want to delete {len(selected_items)} experiment(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            for item in selected_items:
                exp = item.data(0, Qt.UserRole)
                if exp in self.experiments:
                    self.experiments.remove(exp)
                self.experiment_tree.takeTopLevelItem(
                    self.experiment_tree.indexOfTopLevelItem(item)
                )

            self.update_statistics()

    def save_current_experiment(self):
        """Save the current experiment."""
        if not self.current_experiment:
            QMessageBox.warning(
                self,
                "No Experiment",
                "No active experiment to save."
            )
            return

        from PyQt5.QtWidgets import QFileDialog
        import json

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Experiment",
            f"experiment_{self.current_experiment['id']}.json",
            "JSON Files (*.json)"
        )

        if filename:
            try:
                # Prepare data for JSON serialization
                save_data = self.current_experiment.copy()
                save_data["start_time"] = save_data["start_time"].isoformat()
                if "end_time" in save_data:
                    save_data["end_time"] = save_data["end_time"].isoformat()

                with open(filename, 'w') as f:
                    json.dump(save_data, f, indent=2)

                QMessageBox.information(
                    self,
                    "Save Successful",
                    f"Experiment saved to {filename}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save experiment: {str(e)}"
                )

    def add_tag_to_current(self):
        """Add a tag to the current experiment."""
        if not self.current_experiment:
            return

        from PyQt5.QtWidgets import QInputDialog
        tag, ok = QInputDialog.getText(
            self,
            "Add Tag",
            "Enter tag:"
        )

        if ok and tag:
            if "tags" not in self.current_experiment:
                self.current_experiment["tags"] = []
            self.current_experiment["tags"].append(tag)

    def add_notes_to_current(self):
        """Add notes to the current experiment."""
        if not self.current_experiment:
            return

        from PyQt5.QtWidgets import QInputDialog
        notes, ok = QInputDialog.getMultiLineText(
            self,
            "Add Notes",
            "Enter notes:",
            self.current_experiment.get("notes", "")
        )

        if ok:
            self.current_experiment["notes"] = notes

    def on_experiment_double_clicked(self, item, column):
        """Handle double-click on experiment."""
        exp = item.data(0, Qt.UserRole)
        if exp:
            self.experiment_selected.emit(exp)

    def show_context_menu(self, position):
        """Show context menu for experiment tree."""
        item = self.experiment_tree.itemAt(position)
        if not item:
            return

        menu = QMenu(self)

        view_action = menu.addAction("View Details")
        export_action = menu.addAction("Export")
        menu.addSeparator()
        delete_action = menu.addAction("Delete")

        action = menu.exec_(self.experiment_tree.mapToGlobal(position))

        if action == view_action:
            exp = item.data(0, Qt.UserRole)
            self.experiment_selected.emit(exp)
        elif action == export_action:
            self.export_experiment(item.data(0, Qt.UserRole))
        elif action == delete_action:
            self.delete_experiment(item)

    def export_experiment(self, experiment: Dict[str, Any]):
        """Export a single experiment."""
        from PyQt5.QtWidgets import QFileDialog
        import json

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Experiment",
            f"experiment_{experiment['id']}.json",
            "JSON Files (*.json)"
        )

        if filename:
            try:
                # Prepare data for JSON serialization
                save_data = experiment.copy()
                save_data["start_time"] = save_data["start_time"].isoformat()
                if "end_time" in save_data:
                    save_data["end_time"] = save_data["end_time"].isoformat()

                with open(filename, 'w') as f:
                    json.dump(save_data, f, indent=2)

                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Experiment exported to {filename}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to export experiment: {str(e)}"
                )

    def delete_experiment(self, item):
        """Delete a single experiment."""
        reply = QMessageBox.question(
            self,
            "Delete Experiment",
            "Are you sure you want to delete this experiment?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            exp = item.data(0, Qt.UserRole)
            if exp in self.experiments:
                self.experiments.remove(exp)
            self.experiment_tree.takeTopLevelItem(
                self.experiment_tree.indexOfTopLevelItem(item)
            )
            self.update_statistics()

    def add_log(self, message: str, level: str = "INFO"):
        """Add a log message (placeholder for integration)."""
        print(f"[{level}] {message}")