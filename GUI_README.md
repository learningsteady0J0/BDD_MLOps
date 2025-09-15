# Vision AI Training Studio - GUI Application

A comprehensive PyQt-based graphical user interface for training and managing Vision AI models with PyTorch Lightning.

## Features

### 🎯 Core Features
- **Model Configuration**
  - Support for ResNet, VGG, and EfficientNet architectures
  - Easy model variant selection (ResNet-18/34/50/101/152, VGG-11/13/16/19, EfficientNet-B0 to B7)
  - Pretrained weights and backbone freezing options
  - Advanced settings for dropout, label smoothing, and stochastic depth

- **Dataset Management**
  - Built-in support for CIFAR-10 and CIFAR-100
  - Custom dataset integration
  - Data augmentation presets (None, Basic, Standard, Advanced, Custom)
  - Real-time dataset preview and statistics

- **Training Control**
  - Start/Stop/Pause training controls
  - Optimizer selection (Adam, AdamW, SGD, RMSprop)
  - Learning rate schedulers (StepLR, CosineAnnealing, ReduceLROnPlateau, etc.)
  - Early stopping configuration
  - Checkpoint management

- **Real-time Monitoring**
  - Live training progress tracking
  - Epoch and batch progress bars
  - Current metrics display (loss, accuracy, learning rate)
  - Time elapsed and estimated time remaining

### 📊 Visualization
- **Metrics Plots**
  - Real-time loss curves (train/validation)
  - Accuracy tracking
  - Learning rate visualization
  - Combined metrics view
  - Smoothing and log-scale options
  - Export to CSV functionality

- **Experiment Tracking**
  - Experiment history management
  - Compare multiple experiments
  - Export/import experiment configurations
  - Add tags and notes to experiments

### 📝 Logging
- **Comprehensive Log Viewer**
  - Color-coded log levels
  - Search and filter capabilities
  - Auto-scroll option
  - Save logs to file
  - Clear and export functions

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for GPU acceleration)

### Install Dependencies

```bash
# Install GUI-specific requirements
pip install -r requirements_gui.txt

# Or install all requirements including core dependencies
pip install -r requirements.txt
pip install -r requirements_gui.txt
```

## Usage

### Starting the Application

```bash
# Run the GUI application
python gui_app.py
```

### Quick Start Guide

1. **Configure Model**
   - Select model type (ResNet/VGG/EfficientNet)
   - Choose model variant
   - Set number of classes
   - Configure advanced settings if needed

2. **Setup Dataset**
   - Select dataset (CIFAR-10/CIFAR-100/Custom)
   - Set data directory path
   - Configure batch size and workers
   - Choose augmentation preset

3. **Configure Training**
   - Set optimizer and learning rate
   - Configure scheduler
   - Set maximum epochs
   - Enable early stopping if desired

4. **Start Training**
   - Click "Start Training" button
   - Monitor progress in real-time
   - View metrics in visualization tab
   - Check logs for detailed information

### Configuration Management

#### Save Configuration
1. Go to File → Save Configuration
2. Choose location and format (JSON/YAML)
3. Configuration is saved for future use

#### Load Configuration
1. Go to File → Open Configuration
2. Select saved configuration file
3. All settings are restored

#### Recent Configurations
- Access recently used configurations from File → Recent Configurations

### Experiment Management

#### View Experiment History
- All experiments are tracked in the Experiment Tracker panel
- Double-click an experiment to view details
- Right-click for context menu options

#### Compare Experiments
1. Select multiple experiments (Ctrl+Click)
2. Click "Compare Selected"
3. View side-by-side comparison

#### Export Experiment
1. Right-click on an experiment
2. Select "Export"
3. Save as JSON file

### Advanced Features

#### Settings Dialog
Access via Edit → Settings to configure:
- Default paths
- Hardware preferences
- Interface appearance
- Logging options
- MLflow integration

#### Keyboard Shortcuts
- `F5` - Start Training
- `Shift+F5` - Stop Training
- `Ctrl+N` - New Configuration
- `Ctrl+O` - Open Configuration
- `Ctrl+S` - Save Configuration
- `Ctrl+Q` - Exit Application

## Architecture

### Component Structure

```
gui/
├── main_window.py          # Main application window
├── widgets/                # GUI widgets
│   ├── model_config_widget.py
│   ├── dataset_config_widget.py
│   ├── training_control_widget.py
│   ├── metrics_visualization_widget.py
│   ├── log_viewer_widget.py
│   └── experiment_tracker_widget.py
├── threads/                # Background threads
│   └── training_thread.py
├── dialogs/                # Dialog windows
│   ├── settings_dialog.py
│   └── about_dialog.py
└── utils/                  # Utility modules
    └── config_manager.py
```

### Integration with Backend

The GUI seamlessly integrates with the existing PyTorch Lightning training framework:

1. **Configuration Translation**: GUI settings are converted to Hydra configuration format
2. **Process Management**: Training runs in a separate process for stability
3. **Real-time Communication**: Output parsing for live metrics updates
4. **Error Handling**: Graceful error recovery and user notifications

## Customization

### Adding New Models

To add support for new model architectures:

1. Register the model in `src/vision/models/`
2. Update `model_config_widget.py` to include the new model
3. Add model-specific parameters to the configuration

### Custom Datasets

To integrate custom datasets:

1. Implement dataset class in `src/vision/data/`
2. Register with DataModuleRegistry
3. Update `dataset_config_widget.py` for GUI support

### Theme Customization

The application supports theme customization:
- Dark theme (default)
- Light theme
- Custom themes via QSS stylesheets

## Troubleshooting

### Common Issues

1. **ImportError for PyQt5**
   ```bash
   pip install PyQt5 --upgrade
   ```

2. **Training doesn't start**
   - Check configuration validity
   - Ensure data directory exists
   - Verify CUDA availability if using GPU

3. **Plots not updating**
   - Enable "Auto Update" in visualization tab
   - Check plot update interval in settings

4. **Memory issues**
   - Reduce batch size
   - Decrease number of workers
   - Enable gradient accumulation

### Debug Mode

Run with debug logging:
```bash
python gui_app.py --debug
```

## Performance Tips

1. **GPU Acceleration**
   - Ensure CUDA is properly installed
   - Use mixed precision training (16-bit)
   - Enable pin memory for faster data transfer

2. **GUI Responsiveness**
   - Training runs in separate thread
   - Adjust plot update intervals
   - Limit log viewer history

3. **Resource Management**
   - Close unused experiment tabs
   - Clear old logs periodically
   - Remove old checkpoints

## Support

For issues, questions, or feature requests:
- Check the documentation
- Review existing configurations
- Contact the development team

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with:
- PyQt5 - Cross-platform GUI framework
- PyTorch Lightning - Deep learning framework
- MLflow - Experiment tracking
- Matplotlib - Visualization library