"""About dialog for the application."""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextBrowser, QTabWidget, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont


class AboutDialog(QDialog):
    """Dialog showing application information."""

    def __init__(self, parent=None):
        """Initialize the about dialog."""
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("About Vision AI Training Studio")
        self.setFixedSize(600, 400)

        layout = QVBoxLayout()

        # Header with logo and title
        header_layout = QHBoxLayout()

        # Application info
        info_layout = QVBoxLayout()

        title_label = QLabel("Vision AI Training Studio")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        info_layout.addWidget(title_label)

        version_label = QLabel("Version 1.0.0")
        version_label.setStyleSheet("color: #666666;")
        info_layout.addWidget(version_label)

        info_layout.addSpacing(10)

        description_label = QLabel(
            "A comprehensive GUI application for training and managing\n"
            "Vision AI models with PyTorch Lightning."
        )
        info_layout.addWidget(description_label)

        header_layout.addLayout(info_layout)
        header_layout.addStretch()

        layout.addLayout(header_layout)
        layout.addSpacing(20)

        # Tab widget for different information
        tabs = QTabWidget()

        # About tab
        about_tab = QWidget()
        about_layout = QVBoxLayout()

        about_text = QTextBrowser()
        about_text.setOpenExternalLinks(True)
        about_text.setHtml("""
            <h3>Vision AI Training Studio</h3>
            <p>
            Vision AI Training Studio is a powerful GUI application designed to simplify
            the process of training computer vision models using PyTorch Lightning.
            </p>

            <h4>Features:</h4>
            <ul>
                <li>Support for multiple model architectures (ResNet, VGG, EfficientNet)</li>
                <li>Built-in dataset support (CIFAR-10, CIFAR-100, custom datasets)</li>
                <li>Real-time training metrics visualization</li>
                <li>Experiment tracking and comparison</li>
                <li>Comprehensive logging and monitoring</li>
                <li>MLflow integration for experiment management</li>
                <li>GPU acceleration support</li>
                <li>Mixed precision training</li>
            </ul>

            <h4>Built with:</h4>
            <ul>
                <li>PyTorch Lightning - Deep learning framework</li>
                <li>PyQt5 - GUI framework</li>
                <li>MLflow - Experiment tracking</li>
                <li>Hydra - Configuration management</li>
                <li>Matplotlib - Visualization</li>
            </ul>
        """)
        about_layout.addWidget(about_text)
        about_tab.setLayout(about_layout)

        # Authors tab
        authors_tab = QWidget()
        authors_layout = QVBoxLayout()

        authors_text = QTextBrowser()
        authors_text.setHtml("""
            <h3>Development Team</h3>

            <h4>Lead Developer</h4>
            <p>Vision AI Team</p>

            <h4>Contributors</h4>
            <ul>
                <li>PyQt GUI Developer - GUI Implementation</li>
                <li>Model Engineer - Model Architecture</li>
                <li>Data Engineer - Data Pipeline</li>
                <li>MLOps Engineer - Training Infrastructure</li>
            </ul>

            <h4>Special Thanks</h4>
            <p>
            Thanks to the open-source community and all contributors to the
            libraries and frameworks used in this project.
            </p>
        """)
        authors_layout.addWidget(authors_text)
        authors_tab.setLayout(authors_layout)

        # License tab
        license_tab = QWidget()
        license_layout = QVBoxLayout()

        license_text = QTextBrowser()
        license_text.setHtml("""
            <h3>License</h3>

            <p>
            MIT License
            </p>

            <p>
            Copyright (c) 2024 Vision AI Team
            </p>

            <p>
            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:
            </p>

            <p>
            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.
            </p>

            <p>
            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
            </p>
        """)
        license_layout.addWidget(license_text)
        license_tab.setLayout(license_layout)

        # System Info tab
        system_tab = QWidget()
        system_layout = QVBoxLayout()

        system_text = QTextBrowser()

        # Get system information
        import sys
        import platform
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
                gpu_name = torch.cuda.get_device_name(0)
            else:
                cuda_version = "N/A"
                gpu_name = "N/A"
        except:
            torch_version = "Not installed"
            cuda_available = False
            cuda_version = "N/A"
            gpu_name = "N/A"

        try:
            from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
            qt_version = QT_VERSION_STR
            pyqt_version = PYQT_VERSION_STR
        except:
            qt_version = "Unknown"
            pyqt_version = "Unknown"

        system_text.setHtml(f"""
            <h3>System Information</h3>

            <h4>Platform</h4>
            <ul>
                <li>OS: {platform.system()} {platform.release()}</li>
                <li>Architecture: {platform.machine()}</li>
                <li>Python: {sys.version.split()[0]}</li>
            </ul>

            <h4>PyQt</h4>
            <ul>
                <li>Qt Version: {qt_version}</li>
                <li>PyQt Version: {pyqt_version}</li>
            </ul>

            <h4>PyTorch</h4>
            <ul>
                <li>Version: {torch_version}</li>
                <li>CUDA Available: {'Yes' if cuda_available else 'No'}</li>
                <li>CUDA Version: {cuda_version}</li>
                <li>GPU: {gpu_name}</li>
            </ul>

            <h4>Paths</h4>
            <ul>
                <li>Python Executable: {sys.executable}</li>
                <li>Working Directory: {platform.os.getcwd()}</li>
            </ul>
        """)
        system_layout.addWidget(system_text)
        system_tab.setLayout(system_layout)

        # Add tabs
        tabs.addTab(about_tab, "About")
        tabs.addTab(authors_tab, "Authors")
        tabs.addTab(license_tab, "License")
        tabs.addTab(system_tab, "System Info")

        layout.addWidget(tabs)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)