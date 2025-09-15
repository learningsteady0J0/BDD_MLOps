"""Widget for viewing training logs."""

from typing import Optional
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QComboBox, QLabel, QCheckBox,
    QLineEdit
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor, QFont


class LogViewerWidget(QWidget):
    """Widget for viewing and filtering training logs."""

    def __init__(self, parent=None):
        """Initialize the log viewer widget."""
        super().__init__(parent)

        self.log_levels = {
            "DEBUG": QColor("#808080"),
            "INFO": QColor("#2196F3"),
            "WARNING": QColor("#FF9800"),
            "ERROR": QColor("#F44336"),
            "CRITICAL": QColor("#B71C1C"),
            "SUCCESS": QColor("#4CAF50")
        }

        self.max_lines = 10000
        self.auto_scroll = True

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Controls
        controls_layout = QHBoxLayout()

        # Log level filter
        controls_layout.addWidget(QLabel("Level:"))
        self.level_combo = QComboBox()
        self.level_combo.addItems(["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_combo.setCurrentText("INFO")
        controls_layout.addWidget(self.level_combo)

        # Search
        controls_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Enter search term...")
        controls_layout.addWidget(self.search_edit)

        self.search_button = QPushButton("Search")
        controls_layout.addWidget(self.search_button)

        # Options
        self.auto_scroll_check = QCheckBox("Auto Scroll")
        self.auto_scroll_check.setChecked(True)
        controls_layout.addWidget(self.auto_scroll_check)

        self.word_wrap_check = QCheckBox("Word Wrap")
        self.word_wrap_check.setChecked(True)
        controls_layout.addWidget(self.word_wrap_check)

        controls_layout.addStretch()

        # Actions
        self.clear_button = QPushButton("Clear")
        controls_layout.addWidget(self.clear_button)

        self.save_button = QPushButton("Save Log")
        controls_layout.addWidget(self.save_button)

        layout.addLayout(controls_layout)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #444444;
            }
        """)
        layout.addWidget(self.log_text)

        # Status bar
        status_layout = QHBoxLayout()
        self.line_count_label = QLabel("Lines: 0")
        status_layout.addWidget(self.line_count_label)

        self.last_update_label = QLabel("Last Update: Never")
        status_layout.addWidget(self.last_update_label)

        status_layout.addStretch()

        layout.addLayout(status_layout)

        self.setLayout(layout)

    def setup_connections(self):
        """Setup signal-slot connections."""
        self.level_combo.currentIndexChanged.connect(self.filter_logs)
        self.search_button.clicked.connect(self.search_logs)
        self.search_edit.returnPressed.connect(self.search_logs)
        self.auto_scroll_check.stateChanged.connect(self.on_auto_scroll_changed)
        self.word_wrap_check.stateChanged.connect(self.on_word_wrap_changed)
        self.clear_button.clicked.connect(self.clear_logs)
        self.save_button.clicked.connect(self.save_logs)

    def add_log(self, message: str, level: str = "INFO", timestamp: Optional[datetime] = None):
        """Add a log message."""
        if timestamp is None:
            timestamp = datetime.now()

        # Format timestamp
        time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]

        # Create formatted log entry
        log_entry = f"[{time_str}] [{level:8}] {message}"

        # Check if we should display this log level
        current_filter = self.level_combo.currentText()
        if current_filter != "All":
            level_priority = {
                "DEBUG": 0, "INFO": 1, "WARNING": 2,
                "ERROR": 3, "CRITICAL": 4, "SUCCESS": 1
            }
            filter_priority = level_priority.get(current_filter, 0)
            message_priority = level_priority.get(level, 0)

            if message_priority < filter_priority:
                return

        # Move cursor to end
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Set format for this log entry
        format = QTextCharFormat()
        format.setForeground(self.log_levels.get(level, QColor("#ffffff")))

        # Insert the log entry
        cursor.insertText(log_entry + "\n", format)

        # Limit the number of lines
        document = self.log_text.document()
        if document.lineCount() > self.max_lines:
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 100)
            cursor.removeSelectedText()

        # Auto scroll if enabled
        if self.auto_scroll:
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )

        # Update status
        self.update_status()

    def filter_logs(self):
        """Filter logs by level."""
        # This would require storing all logs and re-displaying them
        # For simplicity, we'll just clear and note that future logs will be filtered
        pass

    def search_logs(self):
        """Search for text in logs."""
        search_term = self.search_edit.text()
        if not search_term:
            return

        # Clear previous highlights
        cursor = self.log_text.textCursor()
        cursor.select(QTextCursor.Document)
        format = QTextCharFormat()
        format.setBackground(QColor("transparent"))
        cursor.mergeCharFormat(format)

        # Search and highlight
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.Start)

        format = QTextCharFormat()
        format.setBackground(QColor("#FFEB3B"))
        format.setForeground(QColor("#000000"))

        found_count = 0
        while self.log_text.find(search_term):
            cursor = self.log_text.textCursor()
            cursor.mergeCharFormat(format)
            found_count += 1

        # Move to first occurrence
        if found_count > 0:
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.Start)
            self.log_text.find(search_term)

        # Update status
        if found_count > 0:
            self.add_log(f"Found {found_count} occurrences of '{search_term}'", "INFO")
        else:
            self.add_log(f"No occurrences of '{search_term}' found", "WARNING")

    def on_auto_scroll_changed(self):
        """Handle auto scroll checkbox change."""
        self.auto_scroll = self.auto_scroll_check.isChecked()

    def on_word_wrap_changed(self):
        """Handle word wrap checkbox change."""
        if self.word_wrap_check.isChecked():
            self.log_text.setLineWrapMode(QTextEdit.WidgetWidth)
        else:
            self.log_text.setLineWrapMode(QTextEdit.NoWrap)

    def clear_logs(self):
        """Clear all logs."""
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "Clear Logs",
            "Are you sure you want to clear all logs?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.log_text.clear()
            self.update_status()
            self.add_log("Logs cleared", "INFO")

    def save_logs(self):
        """Save logs to file."""
        from PyQt5.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Logs",
            f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*.*)"
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.toPlainText())
                self.add_log(f"Logs saved to {filename}", "SUCCESS")
            except Exception as e:
                self.add_log(f"Failed to save logs: {str(e)}", "ERROR")

    def update_status(self):
        """Update status labels."""
        # Update line count
        line_count = self.log_text.document().lineCount()
        self.line_count_label.setText(f"Lines: {line_count}")

        # Update last update time
        self.last_update_label.setText(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")

    def add_separator(self, char: str = "=", length: int = 80):
        """Add a separator line."""
        self.add_log(char * length, "INFO")