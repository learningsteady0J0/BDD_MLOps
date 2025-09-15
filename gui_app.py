"""Main entry point for Vision AI Training GUI Application."""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor

from gui.main_window import VisionTrainingMainWindow


class SplashScreen(QSplashScreen):
    """Custom splash screen for the application."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.init_ui()

    def init_ui(self):
        """Initialize splash screen UI."""
        # Create a simple splash screen with text
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor("#2196F3"))
        self.setPixmap(pixmap)

        # Add loading message
        self.showMessage(
            "Vision AI Training Studio\n\nLoading...",
            Qt.AlignCenter | Qt.AlignBottom,
            Qt.white
        )

        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.setFont(font)


def setup_application_style(app: QApplication):
    """Setup application style and theme."""
    # Set application style
    app.setStyle("Fusion")

    # Create dark palette
    dark_palette = QPalette()

    # Window colors
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)

    # Base colors
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))

    # Text colors
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)

    # Button colors
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)

    # Bright text
    dark_palette.setColor(QPalette.BrightText, Qt.red)

    # Link colors
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.LinkVisited, QColor(42, 130, 218))

    # Highlight colors
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    # Disabled colors
    dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))

    # Apply palette
    app.setPalette(dark_palette)

    # Additional styling
    app.setStyleSheet("""
        QToolTip {
            color: #ffffff;
            background-color: #2b2b2b;
            border: 1px solid #666666;
            padding: 5px;
            border-radius: 3px;
        }

        QGroupBox {
            font-weight: bold;
            border: 2px solid #555555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }

        QPushButton {
            padding: 5px 15px;
            border-radius: 3px;
            border: 1px solid #555555;
        }

        QPushButton:hover {
            background-color: #484848;
            border: 1px solid #777777;
        }

        QPushButton:pressed {
            background-color: #323232;
        }

        QComboBox {
            padding: 5px;
            border-radius: 3px;
            border: 1px solid #555555;
        }

        QLineEdit, QSpinBox, QDoubleSpinBox {
            padding: 5px;
            border-radius: 3px;
            border: 1px solid #555555;
            background-color: #2b2b2b;
        }

        QTabWidget::pane {
            border: 1px solid #555555;
            background-color: #353535;
        }

        QTabBar::tab {
            padding: 8px 16px;
            margin-right: 2px;
            background-color: #2b2b2b;
            border: 1px solid #555555;
        }

        QTabBar::tab:selected {
            background-color: #353535;
            border-bottom: 2px solid #2196F3;
        }

        QTableWidget {
            gridline-color: #555555;
            background-color: #2b2b2b;
        }

        QTableWidget::item {
            padding: 5px;
        }

        QHeaderView::section {
            background-color: #353535;
            padding: 5px;
            border: 1px solid #555555;
        }

        QTreeWidget {
            background-color: #2b2b2b;
            alternate-background-color: #353535;
        }

        QTreeWidget::item {
            padding: 3px;
        }

        QTreeWidget::item:selected {
            background-color: #2196F3;
        }

        QProgressBar {
            border: 1px solid #555555;
            border-radius: 3px;
            text-align: center;
            background-color: #2b2b2b;
        }

        QProgressBar::chunk {
            background-color: #2196F3;
            border-radius: 2px;
        }

        QMenuBar {
            background-color: #353535;
            padding: 5px;
        }

        QMenuBar::item {
            padding: 5px 10px;
            background-color: transparent;
        }

        QMenuBar::item:selected {
            background-color: #2196F3;
        }

        QMenu {
            background-color: #353535;
            border: 1px solid #555555;
        }

        QMenu::item {
            padding: 5px 20px;
        }

        QMenu::item:selected {
            background-color: #2196F3;
        }

        QStatusBar {
            background-color: #353535;
            border-top: 1px solid #555555;
        }

        QDockWidget {
            color: white;
        }

        QDockWidget::title {
            background-color: #353535;
            padding: 5px;
            border: 1px solid #555555;
        }

        QScrollBar:vertical {
            background-color: #2b2b2b;
            width: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical {
            background-color: #555555;
            min-height: 20px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #777777;
        }

        QScrollBar:horizontal {
            background-color: #2b2b2b;
            height: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:horizontal {
            background-color: #555555;
            min-width: 20px;
            border-radius: 6px;
        }

        QScrollBar::handle:horizontal:hover {
            background-color: #777777;
        }
    """)


def main():
    """Main entry point for the application."""
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Vision AI Training Studio")
    app.setOrganizationName("VisionAI")

    # Setup application style
    setup_application_style(app)

    # Show splash screen
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # Create main window
    window = VisionTrainingMainWindow()

    # Hide splash and show main window after a short delay
    def show_main_window():
        splash.close()
        window.show()

    QTimer.singleShot(2000, show_main_window)

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()