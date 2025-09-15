"""Widget for metrics visualization."""

from typing import Dict, Any, List, Optional
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QCheckBox,
    QTabWidget, QGridLayout, QSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter
)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class MetricsVisualizationWidget(QWidget):
    """Widget for visualizing training metrics."""

    def __init__(self, parent=None):
        """Initialize the metrics visualization widget."""
        super().__init__(parent)

        # Data storage
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": [],
            "epoch": []
        }

        self.max_points = 1000  # Maximum points to keep in memory
        self.auto_update = True

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Create tabs
        self.tabs = QTabWidget()

        # Loss Plot Tab
        loss_tab = QWidget()
        loss_layout = QVBoxLayout(loss_tab)

        # Loss plot controls
        loss_controls_layout = QHBoxLayout()

        loss_controls_layout.addWidget(QLabel("Display:"))
        self.loss_display_combo = QComboBox()
        self.loss_display_combo.addItems(["Both", "Train Only", "Validation Only"])
        loss_controls_layout.addWidget(self.loss_display_combo)

        self.loss_log_scale_check = QCheckBox("Log Scale")
        loss_controls_layout.addWidget(self.loss_log_scale_check)

        self.loss_smooth_check = QCheckBox("Smooth")
        loss_controls_layout.addWidget(self.loss_smooth_check)

        loss_controls_layout.addWidget(QLabel("Smooth Window:"))
        self.loss_smooth_window_spin = QSpinBox()
        self.loss_smooth_window_spin.setRange(1, 50)
        self.loss_smooth_window_spin.setValue(5)
        self.loss_smooth_window_spin.setEnabled(False)
        loss_controls_layout.addWidget(self.loss_smooth_window_spin)

        loss_controls_layout.addStretch()

        self.loss_reset_button = QPushButton("Reset Zoom")
        loss_controls_layout.addWidget(self.loss_reset_button)

        loss_layout.addLayout(loss_controls_layout)

        # Loss plot
        self.loss_figure = Figure(figsize=(10, 6))
        self.loss_canvas = FigureCanvas(self.loss_figure)
        self.loss_toolbar = NavigationToolbar(self.loss_canvas, self)
        loss_layout.addWidget(self.loss_toolbar)
        loss_layout.addWidget(self.loss_canvas)

        # Accuracy Plot Tab
        acc_tab = QWidget()
        acc_layout = QVBoxLayout(acc_tab)

        # Accuracy plot controls
        acc_controls_layout = QHBoxLayout()

        acc_controls_layout.addWidget(QLabel("Display:"))
        self.acc_display_combo = QComboBox()
        self.acc_display_combo.addItems(["Both", "Train Only", "Validation Only"])
        acc_controls_layout.addWidget(self.acc_display_combo)

        self.acc_smooth_check = QCheckBox("Smooth")
        acc_controls_layout.addWidget(self.acc_smooth_check)

        acc_controls_layout.addWidget(QLabel("Smooth Window:"))
        self.acc_smooth_window_spin = QSpinBox()
        self.acc_smooth_window_spin.setRange(1, 50)
        self.acc_smooth_window_spin.setValue(5)
        self.acc_smooth_window_spin.setEnabled(False)
        acc_controls_layout.addWidget(self.acc_smooth_window_spin)

        acc_controls_layout.addStretch()

        self.acc_reset_button = QPushButton("Reset Zoom")
        acc_controls_layout.addWidget(self.acc_reset_button)

        acc_layout.addLayout(acc_controls_layout)

        # Accuracy plot
        self.acc_figure = Figure(figsize=(10, 6))
        self.acc_canvas = FigureCanvas(self.acc_figure)
        self.acc_toolbar = NavigationToolbar(self.acc_canvas, self)
        acc_layout.addWidget(self.acc_toolbar)
        acc_layout.addWidget(self.acc_canvas)

        # Learning Rate Tab
        lr_tab = QWidget()
        lr_layout = QVBoxLayout(lr_tab)

        # LR plot controls
        lr_controls_layout = QHBoxLayout()

        self.lr_log_scale_check = QCheckBox("Log Scale")
        lr_controls_layout.addWidget(self.lr_log_scale_check)

        lr_controls_layout.addStretch()

        self.lr_reset_button = QPushButton("Reset Zoom")
        lr_controls_layout.addWidget(self.lr_reset_button)

        lr_layout.addLayout(lr_controls_layout)

        # Learning rate plot
        self.lr_figure = Figure(figsize=(10, 6))
        self.lr_canvas = FigureCanvas(self.lr_figure)
        self.lr_toolbar = NavigationToolbar(self.lr_canvas, self)
        lr_layout.addWidget(self.lr_toolbar)
        lr_layout.addWidget(self.lr_canvas)

        # Combined Metrics Tab
        combined_tab = QWidget()
        combined_layout = QVBoxLayout(combined_tab)

        # Combined plot controls
        combined_controls_layout = QHBoxLayout()

        self.auto_update_check = QCheckBox("Auto Update")
        self.auto_update_check.setChecked(True)
        combined_controls_layout.addWidget(self.auto_update_check)

        combined_controls_layout.addStretch()

        self.export_button = QPushButton("Export Data")
        combined_controls_layout.addWidget(self.export_button)

        self.clear_button = QPushButton("Clear All")
        combined_controls_layout.addWidget(self.clear_button)

        combined_layout.addLayout(combined_controls_layout)

        # Combined plot with subplots
        self.combined_figure = Figure(figsize=(12, 8))
        self.combined_canvas = FigureCanvas(self.combined_figure)
        combined_layout.addWidget(self.combined_canvas)

        # Statistics Tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        # Statistics table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(5)
        self.stats_table.setHorizontalHeaderLabels([
            "Metric", "Current", "Best", "Worst", "Average"
        ])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setAlternatingRowColors(True)

        # Initialize rows
        metrics = [
            "Train Loss", "Val Loss", "Train Accuracy", "Val Accuracy",
            "Learning Rate", "Epoch Time", "Best Epoch"
        ]
        self.stats_table.setRowCount(len(metrics))
        for i, metric in enumerate(metrics):
            self.stats_table.setItem(i, 0, QTableWidgetItem(metric))
            for j in range(1, 5):
                self.stats_table.setItem(i, j, QTableWidgetItem("N/A"))

        stats_layout.addWidget(self.stats_table)

        # Summary statistics
        summary_group = QGroupBox("Training Summary")
        summary_layout = QGridLayout()

        summary_layout.addWidget(QLabel("Total Epochs:"), 0, 0)
        self.total_epochs_label = QLabel("0")
        summary_layout.addWidget(self.total_epochs_label, 0, 1)

        summary_layout.addWidget(QLabel("Total Time:"), 1, 0)
        self.total_time_label = QLabel("00:00:00")
        summary_layout.addWidget(self.total_time_label, 1, 1)

        summary_layout.addWidget(QLabel("Avg Epoch Time:"), 2, 0)
        self.avg_epoch_time_label = QLabel("00:00:00")
        summary_layout.addWidget(self.avg_epoch_time_label, 2, 1)

        summary_layout.addWidget(QLabel("Best Val Loss:"), 3, 0)
        self.best_val_loss_label = QLabel("N/A")
        self.best_val_loss_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        summary_layout.addWidget(self.best_val_loss_label, 3, 1)

        summary_layout.addWidget(QLabel("Best Val Acc:"), 4, 0)
        self.best_val_acc_label = QLabel("N/A")
        self.best_val_acc_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        summary_layout.addWidget(self.best_val_acc_label, 4, 1)

        summary_group.setLayout(summary_layout)
        stats_layout.addWidget(summary_group)

        # Add tabs
        self.tabs.addTab(loss_tab, "Loss")
        self.tabs.addTab(acc_tab, "Accuracy")
        self.tabs.addTab(lr_tab, "Learning Rate")
        self.tabs.addTab(combined_tab, "Combined")
        self.tabs.addTab(stats_tab, "Statistics")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

        # Initialize plots
        self.init_plots()

    def init_plots(self):
        """Initialize all plots."""
        # Loss plot
        self.loss_ax = self.loss_figure.add_subplot(111)
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.set_title("Training and Validation Loss")
        self.loss_ax.grid(True, alpha=0.3)

        # Accuracy plot
        self.acc_ax = self.acc_figure.add_subplot(111)
        self.acc_ax.set_xlabel("Epoch")
        self.acc_ax.set_ylabel("Accuracy (%)")
        self.acc_ax.set_title("Training and Validation Accuracy")
        self.acc_ax.grid(True, alpha=0.3)

        # Learning rate plot
        self.lr_ax = self.lr_figure.add_subplot(111)
        self.lr_ax.set_xlabel("Epoch")
        self.lr_ax.set_ylabel("Learning Rate")
        self.lr_ax.set_title("Learning Rate Schedule")
        self.lr_ax.grid(True, alpha=0.3)

        # Combined plots
        self.combined_axes = []
        for i in range(4):
            ax = self.combined_figure.add_subplot(2, 2, i + 1)
            self.combined_axes.append(ax)

        self.combined_axes[0].set_title("Loss")
        self.combined_axes[0].set_ylabel("Loss")
        self.combined_axes[0].grid(True, alpha=0.3)

        self.combined_axes[1].set_title("Accuracy")
        self.combined_axes[1].set_ylabel("Accuracy (%)")
        self.combined_axes[1].grid(True, alpha=0.3)

        self.combined_axes[2].set_title("Learning Rate")
        self.combined_axes[2].set_xlabel("Epoch")
        self.combined_axes[2].set_ylabel("LR")
        self.combined_axes[2].grid(True, alpha=0.3)

        self.combined_axes[3].set_title("Loss vs Accuracy")
        self.combined_axes[3].set_xlabel("Epoch")
        self.combined_axes[3].set_ylabel("Value")
        self.combined_axes[3].grid(True, alpha=0.3)

        self.combined_figure.tight_layout()

    def setup_connections(self):
        """Setup signal-slot connections."""
        # Loss plot controls
        self.loss_display_combo.currentIndexChanged.connect(self.update_loss_plot)
        self.loss_log_scale_check.stateChanged.connect(self.update_loss_plot)
        self.loss_smooth_check.stateChanged.connect(self.on_loss_smooth_changed)
        self.loss_smooth_window_spin.valueChanged.connect(self.update_loss_plot)
        self.loss_reset_button.clicked.connect(self.reset_loss_zoom)

        # Accuracy plot controls
        self.acc_display_combo.currentIndexChanged.connect(self.update_accuracy_plot)
        self.acc_smooth_check.stateChanged.connect(self.on_acc_smooth_changed)
        self.acc_smooth_window_spin.valueChanged.connect(self.update_accuracy_plot)
        self.acc_reset_button.clicked.connect(self.reset_acc_zoom)

        # Learning rate plot controls
        self.lr_log_scale_check.stateChanged.connect(self.update_lr_plot)
        self.lr_reset_button.clicked.connect(self.reset_lr_zoom)

        # Combined controls
        self.auto_update_check.stateChanged.connect(self.on_auto_update_changed)
        self.export_button.clicked.connect(self.export_metrics)
        self.clear_button.clicked.connect(self.clear_all_metrics)

    def on_loss_smooth_changed(self):
        """Handle loss smoothing checkbox change."""
        self.loss_smooth_window_spin.setEnabled(self.loss_smooth_check.isChecked())
        self.update_loss_plot()

    def on_acc_smooth_changed(self):
        """Handle accuracy smoothing checkbox change."""
        self.acc_smooth_window_spin.setEnabled(self.acc_smooth_check.isChecked())
        self.update_accuracy_plot()

    def on_auto_update_changed(self):
        """Handle auto update checkbox change."""
        self.auto_update = self.auto_update_check.isChecked()

    def smooth_data(self, data: List[float], window: int) -> np.ndarray:
        """Apply smoothing to data."""
        if len(data) < window:
            return np.array(data)

        # Use simple moving average
        smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
        # Pad the beginning to maintain array length
        padding = np.array(data[:window-1])
        return np.concatenate([padding, smoothed])

    @pyqtSlot(dict)
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics with new data."""
        # Add to history
        if "epoch" in metrics:
            self.metrics_history["epoch"].append(metrics["epoch"])

            # Ensure we don't exceed max points
            if len(self.metrics_history["epoch"]) > self.max_points:
                for key in self.metrics_history:
                    if self.metrics_history[key]:
                        self.metrics_history[key].pop(0)

        # Update individual metrics
        for key in ["train_loss", "val_loss", "train_acc", "val_acc", "learning_rate"]:
            if key in metrics:
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(metrics[key])

        # Update plots if auto-update is enabled
        if self.auto_update:
            self.update_all_plots()

        # Update statistics
        self.update_statistics(metrics)

    def update_all_plots(self):
        """Update all plots."""
        self.update_loss_plot()
        self.update_accuracy_plot()
        self.update_lr_plot()
        self.update_combined_plot()

    def update_loss_plot(self):
        """Update the loss plot."""
        self.loss_ax.clear()
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.set_title("Training and Validation Loss")
        self.loss_ax.grid(True, alpha=0.3)

        if not self.metrics_history["epoch"]:
            self.loss_canvas.draw()
            return

        epochs = self.metrics_history["epoch"]
        display_mode = self.loss_display_combo.currentText()

        # Plot train loss
        if display_mode in ["Both", "Train Only"] and self.metrics_history["train_loss"]:
            train_loss = self.metrics_history["train_loss"]
            if self.loss_smooth_check.isChecked():
                train_loss = self.smooth_data(train_loss, self.loss_smooth_window_spin.value())
            self.loss_ax.plot(epochs[:len(train_loss)], train_loss,
                            label="Train Loss", color="#2196F3", linewidth=2)

        # Plot validation loss
        if display_mode in ["Both", "Validation Only"] and self.metrics_history["val_loss"]:
            val_loss = self.metrics_history["val_loss"]
            if self.loss_smooth_check.isChecked():
                val_loss = self.smooth_data(val_loss, self.loss_smooth_window_spin.value())
            self.loss_ax.plot(epochs[:len(val_loss)], val_loss,
                            label="Val Loss", color="#FF9800", linewidth=2)

        if self.loss_log_scale_check.isChecked():
            self.loss_ax.set_yscale('log')

        self.loss_ax.legend(loc="upper right")
        self.loss_canvas.draw()

    def update_accuracy_plot(self):
        """Update the accuracy plot."""
        self.acc_ax.clear()
        self.acc_ax.set_xlabel("Epoch")
        self.acc_ax.set_ylabel("Accuracy (%)")
        self.acc_ax.set_title("Training and Validation Accuracy")
        self.acc_ax.grid(True, alpha=0.3)

        if not self.metrics_history["epoch"]:
            self.acc_canvas.draw()
            return

        epochs = self.metrics_history["epoch"]
        display_mode = self.acc_display_combo.currentText()

        # Plot train accuracy
        if display_mode in ["Both", "Train Only"] and self.metrics_history["train_acc"]:
            train_acc = self.metrics_history["train_acc"]
            if self.acc_smooth_check.isChecked():
                train_acc = self.smooth_data(train_acc, self.acc_smooth_window_spin.value())
            self.acc_ax.plot(epochs[:len(train_acc)], train_acc,
                           label="Train Acc", color="#4CAF50", linewidth=2)

        # Plot validation accuracy
        if display_mode in ["Both", "Validation Only"] and self.metrics_history["val_acc"]:
            val_acc = self.metrics_history["val_acc"]
            if self.acc_smooth_check.isChecked():
                val_acc = self.smooth_data(val_acc, self.acc_smooth_window_spin.value())
            self.acc_ax.plot(epochs[:len(val_acc)], val_acc,
                           label="Val Acc", color="#9C27B0", linewidth=2)

        self.acc_ax.set_ylim([0, 100])
        self.acc_ax.legend(loc="lower right")
        self.acc_canvas.draw()

    def update_lr_plot(self):
        """Update the learning rate plot."""
        self.lr_ax.clear()
        self.lr_ax.set_xlabel("Epoch")
        self.lr_ax.set_ylabel("Learning Rate")
        self.lr_ax.set_title("Learning Rate Schedule")
        self.lr_ax.grid(True, alpha=0.3)

        if not self.metrics_history["epoch"] or not self.metrics_history["learning_rate"]:
            self.lr_canvas.draw()
            return

        epochs = self.metrics_history["epoch"]
        lr = self.metrics_history["learning_rate"]

        self.lr_ax.plot(epochs[:len(lr)], lr, color="#E91E63", linewidth=2)

        if self.lr_log_scale_check.isChecked():
            self.lr_ax.set_yscale('log')

        self.lr_canvas.draw()

    def update_combined_plot(self):
        """Update the combined plot."""
        for ax in self.combined_axes:
            ax.clear()

        if not self.metrics_history["epoch"]:
            self.combined_canvas.draw()
            return

        epochs = self.metrics_history["epoch"]

        # Plot 1: Loss
        ax = self.combined_axes[0]
        ax.set_title("Loss")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        if self.metrics_history["train_loss"]:
            ax.plot(epochs[:len(self.metrics_history["train_loss"])],
                   self.metrics_history["train_loss"],
                   label="Train", color="#2196F3")
        if self.metrics_history["val_loss"]:
            ax.plot(epochs[:len(self.metrics_history["val_loss"])],
                   self.metrics_history["val_loss"],
                   label="Val", color="#FF9800")
        ax.legend(loc="upper right", fontsize=8)

        # Plot 2: Accuracy
        ax = self.combined_axes[1]
        ax.set_title("Accuracy")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        if self.metrics_history["train_acc"]:
            ax.plot(epochs[:len(self.metrics_history["train_acc"])],
                   self.metrics_history["train_acc"],
                   label="Train", color="#4CAF50")
        if self.metrics_history["val_acc"]:
            ax.plot(epochs[:len(self.metrics_history["val_acc"])],
                   self.metrics_history["val_acc"],
                   label="Val", color="#9C27B0")
        ax.legend(loc="lower right", fontsize=8)

        # Plot 3: Learning Rate
        ax = self.combined_axes[2]
        ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.grid(True, alpha=0.3)
        if self.metrics_history["learning_rate"]:
            ax.plot(epochs[:len(self.metrics_history["learning_rate"])],
                   self.metrics_history["learning_rate"],
                   color="#E91E63")

        # Plot 4: Combined normalized
        ax = self.combined_axes[3]
        ax.set_title("Normalized Metrics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized Value")
        ax.grid(True, alpha=0.3)

        # Normalize and plot all metrics
        if self.metrics_history["val_loss"]:
            val_loss_norm = np.array(self.metrics_history["val_loss"])
            if len(val_loss_norm) > 0:
                val_loss_norm = (val_loss_norm - val_loss_norm.min()) / (val_loss_norm.max() - val_loss_norm.min() + 1e-8)
                ax.plot(epochs[:len(val_loss_norm)], val_loss_norm,
                       label="Val Loss", color="#FF9800", alpha=0.7)

        if self.metrics_history["val_acc"]:
            val_acc_norm = np.array(self.metrics_history["val_acc"]) / 100.0
            ax.plot(epochs[:len(val_acc_norm)], val_acc_norm,
                   label="Val Acc", color="#9C27B0", alpha=0.7)

        ax.legend(loc="center right", fontsize=8)
        ax.set_ylim([0, 1])

        self.combined_figure.tight_layout()
        self.combined_canvas.draw()

    def update_statistics(self, metrics: Dict[str, Any]):
        """Update statistics table."""
        # Update current values
        if "train_loss" in metrics:
            self.stats_table.item(0, 1).setText(f"{metrics['train_loss']:.4f}")
        if "val_loss" in metrics:
            self.stats_table.item(1, 1).setText(f"{metrics['val_loss']:.4f}")
        if "train_acc" in metrics:
            self.stats_table.item(2, 1).setText(f"{metrics['train_acc']:.2f}%")
        if "val_acc" in metrics:
            self.stats_table.item(3, 1).setText(f"{metrics['val_acc']:.2f}%")
        if "learning_rate" in metrics:
            self.stats_table.item(4, 1).setText(f"{metrics['learning_rate']:.6f}")

        # Calculate and update best/worst/average
        for i, key in enumerate(["train_loss", "val_loss", "train_acc", "val_acc"]):
            if key in self.metrics_history and self.metrics_history[key]:
                values = self.metrics_history[key]
                if "loss" in key:
                    best = min(values)
                    worst = max(values)
                else:  # accuracy
                    best = max(values)
                    worst = min(values)
                avg = np.mean(values)

                # Format based on metric type
                if "acc" in key:
                    self.stats_table.item(i, 2).setText(f"{best:.2f}%")
                    self.stats_table.item(i, 3).setText(f"{worst:.2f}%")
                    self.stats_table.item(i, 4).setText(f"{avg:.2f}%")
                else:
                    self.stats_table.item(i, 2).setText(f"{best:.4f}")
                    self.stats_table.item(i, 3).setText(f"{worst:.4f}")
                    self.stats_table.item(i, 4).setText(f"{avg:.4f}")

        # Update summary
        if self.metrics_history["epoch"]:
            self.total_epochs_label.setText(str(len(self.metrics_history["epoch"])))

        # Update best values
        if self.metrics_history["val_loss"]:
            best_val_loss = min(self.metrics_history["val_loss"])
            self.best_val_loss_label.setText(f"{best_val_loss:.4f}")

        if self.metrics_history["val_acc"]:
            best_val_acc = max(self.metrics_history["val_acc"])
            self.best_val_acc_label.setText(f"{best_val_acc:.2f}%")

    def reset_loss_zoom(self):
        """Reset loss plot zoom."""
        self.loss_ax.autoscale()
        self.loss_canvas.draw()

    def reset_acc_zoom(self):
        """Reset accuracy plot zoom."""
        self.acc_ax.autoscale()
        self.acc_ax.set_ylim([0, 100])
        self.acc_canvas.draw()

    def reset_lr_zoom(self):
        """Reset learning rate plot zoom."""
        self.lr_ax.autoscale()
        self.lr_canvas.draw()

    def export_metrics(self):
        """Export metrics to CSV."""
        from PyQt5.QtWidgets import QFileDialog
        import csv

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Metrics",
            "metrics.csv",
            "CSV Files (*.csv)"
        )

        if filename:
            try:
                with open(filename, 'w', newline='') as csvfile:
                    fieldnames = list(self.metrics_history.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    # Transpose data for row-wise writing
                    max_len = max(len(v) for v in self.metrics_history.values() if v)
                    for i in range(max_len):
                        row = {}
                        for key in fieldnames:
                            if i < len(self.metrics_history[key]):
                                row[key] = self.metrics_history[key][i]
                        writer.writerow(row)

                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "Export Successful",
                                      f"Metrics exported to {filename}")
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Export Error",
                                   f"Failed to export metrics: {str(e)}")

    def clear_all_metrics(self):
        """Clear all metrics data."""
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "Clear Metrics",
            "Are you sure you want to clear all metrics data?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Clear history
            for key in self.metrics_history:
                self.metrics_history[key] = []

            # Clear plots
            self.update_all_plots()

            # Reset statistics
            for i in range(self.stats_table.rowCount()):
                for j in range(1, 5):
                    self.stats_table.item(i, j).setText("N/A")

            # Reset summary
            self.total_epochs_label.setText("0")
            self.total_time_label.setText("00:00:00")
            self.avg_epoch_time_label.setText("00:00:00")
            self.best_val_loss_label.setText("N/A")
            self.best_val_acc_label.setText("N/A")