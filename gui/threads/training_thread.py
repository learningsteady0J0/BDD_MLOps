"""Thread for running training in background."""

import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import subprocess
import json
import yaml

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
import torch


class TrainingThread(QThread):
    """Thread for running training process."""

    # Signals
    training_started = pyqtSignal()
    training_progress = pyqtSignal(dict)
    training_completed = pyqtSignal()
    training_error = pyqtSignal(str)
    log_message = pyqtSignal(str, str)  # message, level
    metrics_update = pyqtSignal(dict)

    def __init__(self, config: Dict[str, Any]):
        """Initialize the training thread."""
        super().__init__()
        self.config = config
        self.is_running = False
        self.is_paused = False
        self.should_stop = False

        # Training state
        self.current_epoch = 0
        self.total_epochs = config.get("training", {}).get("max_epochs", 100)
        self.start_time = None
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

        # Process for running training
        self.training_process = None

    def run(self):
        """Run the training process."""
        try:
            self.is_running = True
            self.should_stop = False
            self.start_time = datetime.now()

            # Emit start signal
            self.training_started.emit()
            self.log_message.emit("Training started", "INFO")

            # Prepare configuration file
            config_file = self.prepare_config_file()

            # Prepare command
            cmd = self.build_training_command(config_file)
            self.log_message.emit(f"Running command: {' '.join(cmd)}", "DEBUG")

            # Start training process
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )

            # Monitor process output
            while True:
                if self.should_stop:
                    self.log_message.emit("Stopping training...", "WARNING")
                    self.training_process.terminate()
                    break

                if self.is_paused:
                    time.sleep(0.1)
                    continue

                # Read output line
                line = self.training_process.stdout.readline()
                if not line:
                    # Check if process has finished
                    if self.training_process.poll() is not None:
                        break
                    continue

                # Parse and process the line
                self.process_output_line(line.strip())

            # Wait for process to complete
            return_code = self.training_process.wait()

            if return_code == 0 and not self.should_stop:
                self.log_message.emit("Training completed successfully", "SUCCESS")
                self.training_completed.emit()
            elif self.should_stop:
                self.log_message.emit("Training stopped by user", "WARNING")
            else:
                error_output = self.training_process.stderr.read()
                self.log_message.emit(f"Training failed with code {return_code}", "ERROR")
                if error_output:
                    self.log_message.emit(f"Error output: {error_output}", "ERROR")
                self.training_error.emit(f"Training failed with code {return_code}")

        except Exception as e:
            error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(error_msg, "CRITICAL")
            self.training_error.emit(str(e))

        finally:
            self.is_running = False
            self.cleanup()

    def prepare_config_file(self) -> str:
        """Prepare configuration file for training."""
        # Create temporary config file
        config_dir = Path("configs/temp")
        config_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = config_dir / f"training_config_{timestamp}.yaml"

        # Convert config to Hydra format
        hydra_config = self.convert_to_hydra_config(self.config)

        # Save config
        with open(config_file, 'w') as f:
            yaml.dump(hydra_config, f, default_flow_style=False)

        self.log_message.emit(f"Configuration saved to {config_file}", "DEBUG")
        return str(config_file)

    def convert_to_hydra_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert GUI config to Hydra config format."""
        hydra_config = {
            "experiment": {
                "name": config.get("experiment", {}).get("name", "gui_experiment"),
                "seed": config.get("experiment", {}).get("seed", 42)
            },
            "model": {
                "name": config.get("model", {}).get("name", "resnet"),
                "num_classes": config.get("model", {}).get("num_classes", 10),
                "pretrained": config.get("model", {}).get("pretrained", True),
                "freeze_backbone": config.get("model", {}).get("freeze_backbone", False),
                "dropout_rate": config.get("model", {}).get("dropout_rate", 0.5)
            },
            "data": {
                "name": config.get("data", {}).get("name", "cifar10"),
                "data_dir": config.get("data", {}).get("data_dir", "./data"),
                "batch_size": config.get("data", {}).get("batch_size", 32),
                "num_workers": config.get("data", {}).get("num_workers", 4),
                "pin_memory": config.get("data", {}).get("pin_memory", True),
                "persistent_workers": config.get("data", {}).get("persistent_workers", True)
            },
            "training": {
                "max_epochs": config.get("training", {}).get("max_epochs", 100),
                "learning_rate": config.get("training", {}).get("learning_rate", 0.001),
                "weight_decay": config.get("training", {}).get("weight_decay", 0.0001),
                "optimizer": config.get("training", {}).get("optimizer", "adamw"),
                "gradient_clip_val": config.get("training", {}).get("gradient_clip_val", 1.0),
                "accumulate_grad_batches": config.get("training", {}).get("accumulate_grad_batches", 1)
            },
            "hardware": {
                "accelerator": "auto",
                "devices": "auto",
                "precision": config.get("training", {}).get("precision", 16),
                "num_workers": config.get("data", {}).get("num_workers", 4)
            }
        }

        # Add model-specific config
        model_name = config.get("model", {}).get("name", "resnet")
        if model_name == "resnet":
            hydra_config["model"]["resnet_version"] = config.get("model", {}).get("variant", "resnet50")
        elif model_name == "vgg":
            hydra_config["model"]["vgg_version"] = config.get("model", {}).get("variant", "vgg16")
        elif model_name == "efficientnet":
            hydra_config["model"]["efficientnet_version"] = config.get("model", {}).get("variant", "efficientnet_b0")

        # Add scheduler config
        scheduler = config.get("training", {}).get("scheduler", {})
        if scheduler.get("type") and scheduler["type"] != "None":
            hydra_config["training"]["scheduler"] = {
                "type": scheduler["type"].lower(),
                "step_size": scheduler.get("step_size", 10),
                "gamma": scheduler.get("gamma", 0.1)
            }

        # Add early stopping if enabled
        early_stopping = config.get("training", {}).get("early_stopping")
        if early_stopping:
            hydra_config["callbacks"] = {
                "early_stopping": {
                    "monitor": early_stopping.get("monitor", "val_loss"),
                    "patience": early_stopping.get("patience", 10),
                    "min_delta": early_stopping.get("min_delta", 0.0001),
                    "mode": early_stopping.get("mode", "min")
                }
            }

        return hydra_config

    def build_training_command(self, config_file: str) -> list:
        """Build the training command."""
        # Get project root
        project_root = Path(__file__).parent.parent.parent

        # Build command
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            "--config-path", str(Path(config_file).parent),
            "--config-name", Path(config_file).stem
        ]

        return cmd

    def process_output_line(self, line: str):
        """Process a line of output from the training process."""
        if not line:
            return

        # Log the raw line
        self.log_message.emit(line, "INFO")

        # Try to parse metrics from the line
        metrics = self.parse_metrics_from_line(line)
        if metrics:
            self.update_progress(metrics)
            self.metrics_update.emit(metrics)

    def parse_metrics_from_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse metrics from output line."""
        metrics = {}

        # Parse epoch information
        if "Epoch" in line:
            try:
                # Try to parse epoch number
                if "Epoch " in line and "/" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Epoch" and i + 1 < len(parts):
                            epoch_str = parts[i + 1]
                            if "/" in epoch_str:
                                current, total = epoch_str.split("/")
                                metrics["epoch"] = int(current)
                                metrics["total_epochs"] = int(total)
                                self.current_epoch = int(current)
            except:
                pass

        # Parse loss values
        if "loss" in line.lower():
            try:
                # Look for patterns like "loss: 0.1234" or "train_loss=0.1234"
                import re

                # Train loss
                train_loss_match = re.search(r'train[_\s]loss[:\s=]+(\d+\.?\d*)', line, re.IGNORECASE)
                if train_loss_match:
                    metrics["train_loss"] = float(train_loss_match.group(1))

                # Val loss
                val_loss_match = re.search(r'val[_\s]loss[:\s=]+(\d+\.?\d*)', line, re.IGNORECASE)
                if val_loss_match:
                    metrics["val_loss"] = float(val_loss_match.group(1))

                    # Update best val loss
                    if metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = metrics["val_loss"]
            except:
                pass

        # Parse accuracy values
        if "acc" in line.lower() or "accuracy" in line.lower():
            try:
                import re

                # Train accuracy
                train_acc_match = re.search(r'train[_\s]acc(?:uracy)?[:\s=]+(\d+\.?\d*)', line, re.IGNORECASE)
                if train_acc_match:
                    metrics["train_acc"] = float(train_acc_match.group(1))
                    # Convert to percentage if needed
                    if metrics["train_acc"] <= 1.0:
                        metrics["train_acc"] *= 100

                # Val accuracy
                val_acc_match = re.search(r'val[_\s]acc(?:uracy)?[:\s=]+(\d+\.?\d*)', line, re.IGNORECASE)
                if val_acc_match:
                    metrics["val_acc"] = float(val_acc_match.group(1))
                    # Convert to percentage if needed
                    if metrics["val_acc"] <= 1.0:
                        metrics["val_acc"] *= 100

                    # Update best val accuracy
                    if metrics["val_acc"] > self.best_val_acc:
                        self.best_val_acc = metrics["val_acc"]
            except:
                pass

        # Parse learning rate
        if "lr" in line.lower() or "learning_rate" in line.lower():
            try:
                import re
                lr_match = re.search(r'(?:lr|learning_rate)[:\s=]+(\d+\.?\d*(?:e[+-]?\d+)?)', line, re.IGNORECASE)
                if lr_match:
                    metrics["learning_rate"] = float(lr_match.group(1))
            except:
                pass

        return metrics if metrics else None

    def update_progress(self, metrics: Dict[str, Any]):
        """Update training progress."""
        progress_data = {}

        # Calculate overall progress
        if self.current_epoch > 0 and self.total_epochs > 0:
            overall_progress = (self.current_epoch / self.total_epochs) * 100
            progress_data["overall_progress"] = overall_progress

        # Add current epoch info
        progress_data["current_epoch"] = self.current_epoch
        progress_data["total_epochs"] = self.total_epochs

        # Calculate time elapsed and remaining
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            progress_data["time_elapsed"] = str(elapsed).split('.')[0]

            # Estimate time remaining
            if self.current_epoch > 0:
                time_per_epoch = elapsed / self.current_epoch
                remaining_epochs = self.total_epochs - self.current_epoch
                time_remaining = time_per_epoch * remaining_epochs
                progress_data["time_remaining"] = str(time_remaining).split('.')[0]

        # Add metrics
        progress_data.update(metrics)

        # Emit progress update
        self.training_progress.emit(progress_data)

    def stop(self):
        """Stop the training process."""
        self.should_stop = True
        if self.training_process:
            try:
                self.training_process.terminate()
                # Give it time to terminate gracefully
                time.sleep(2)
                if self.training_process.poll() is None:
                    # Force kill if still running
                    self.training_process.kill()
            except:
                pass

    def pause(self):
        """Pause the training process."""
        self.is_paused = True
        self.log_message.emit("Training paused", "WARNING")

    def resume(self):
        """Resume the training process."""
        self.is_paused = False
        self.log_message.emit("Training resumed", "INFO")

    def cleanup(self):
        """Clean up resources."""
        if self.training_process:
            try:
                self.training_process.terminate()
            except:
                pass
            self.training_process = None