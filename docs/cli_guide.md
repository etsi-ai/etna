# ETNA CLI Implementation Guide

## Overview

The ETNA CLI (`cli.py`) is a comprehensive command-line interface that integrates all components of the ETNA neural network framework. It provides a unified interface for building, testing, training, and evaluating neural network models.

## Architecture

### Core Components

1. **ETNACLIBuilder**: Handles compilation and building of both Rust and Python components
2. **ETNATestRunner**: Manages testing operations for both Rust and Python code
3. **ETNAModelManager**: Handles neural network training and prediction operations
4. **ETNAMetricsEvaluator**: Provides comprehensive model evaluation capabilities
5. **ETNAProjectManager**: Manages project creation and template generation

### Integration Points

#### Rust Integration
- Uses Cargo for building Rust core (`etna_core/`)
- Interfaces with PyO3 bindings for Python-Rust communication
- Compiles to native extensions for optimal performance

#### Python Integration
- Uses Maturin for building Python packages with Rust extensions
- Integrates with the metrics, preprocessing, and API modules
- Provides seamless Python API access

#### Testing Integration
- Runs Rust tests using `cargo test`
- Runs Python tests using `pytest`
- Includes integration tests that verify Rust-Python interoperability

## File Structure

```
etna/
├── cli.py                  # Main CLI implementation
├── etna_cli.py            # Entry point script
├── etna.ps1               # PowerShell wrapper
├── etna.bat               # Batch file wrapper
├── manage.ps1             # Project management script
├── .vscode/
│   └── tasks.json         # VS Code task definitions
└── examples/
    ├── sample_data.json   # Sample training data
    ├── test_data.json     # Sample test data
    ├── config.json        # Training configuration
    └── ground_truth.json  # Ground truth labels
```

## Command Reference

### Build Commands

#### `etna build`
Builds both Rust core and Python package.

**Implementation:**
- Calls `cargo build` in `etna_core/` directory
- Calls `maturin develop --release` for Python package
- Handles error reporting and validation

**Options:**
- `--release`: Build in optimized release mode
- `--rust-only`: Build only the Rust core
- `--python-only`: Build only the Python package

#### `etna clean`
Removes all build artifacts.

**Implementation:**
- Runs `cargo clean` for Rust artifacts
- Runs `maturin clean` for Python artifacts
- Removes additional build directories (`build/`, `dist/`, etc.)

### Testing Commands

#### `etna test`
Runs comprehensive test suite.

**Implementation:**
- Executes Rust tests via `cargo test`
- Executes Python tests via `pytest`
- Runs integration tests to verify Rust-Python bindings
- Optionally generates coverage reports

**Options:**
- `--rust`: Run only Rust tests
- `--python`: Run only Python tests
- `--integration`: Run only integration tests
- `--coverage`: Generate coverage report

### Model Commands

#### `etna train --data <file>`
Trains neural network models using the Rust core.

**Implementation:**
- Loads training data from JSON file
- Configures model parameters from config file or CLI args
- Calls Rust training functions via PyO3 bindings
- Handles data validation and preprocessing

**Data Format:**
```json
{
  "X": [[feature1, feature2, ...], ...],
  "y": [label1, label2, ...]
}
```

#### `etna predict --data <file>`
Makes predictions using trained models.

**Implementation:**
- Loads test data from JSON file
- Calls Rust prediction functions
- Saves predictions to JSON output file
- Handles model loading and validation

#### `etna metrics --pred <pred_file> --true <true_file>`
Evaluates model performance using comprehensive metrics.

**Implementation:**
- Loads predictions and ground truth from JSON files
- Uses `ClassificationMetrics` or `RegressionMetrics` classes
- Calculates accuracy, precision, recall, F1-score, confusion matrix
- Generates detailed evaluation reports

### Project Management

#### `etna create <project_name>`
Creates new ETNA project from templates.

**Implementation:**
- Creates directory structure with `src/`, `tests/`, `examples/`
- Generates sample training scripts and data files
- Creates project configuration files
- Provides getting-started instructions

#### `etna install`
Installs dependencies and the ETNA package.

**Implementation:**
- Installs Python dependencies via pip
- Installs the package in development mode
- Handles optional development dependencies

## Integration with Rust Components

### Core Neural Network
The CLI integrates with the Rust implementation in `etna_core/src/`:

- **`lib.rs`**: Main PyO3 module with `train()` and `predict()` functions
- **`model.rs`**: Neural network implementation with forward/backward passes
- **`layers.rs`**: Linear layer implementations
- **`optimizer.rs`**: SGD and other optimization algorithms
- **`loss_function.rs`**: Cross-entropy and MSE loss functions

### Data Flow
1. CLI loads JSON data and converts to Python lists
2. PyO3 converts Python data to Rust `Vec<Vec<f32>>`
3. Rust core performs training/prediction
4. Results are converted back to Python and saved as JSON

## Integration with Python Components

### Metrics Module
The CLI uses the comprehensive metrics implementation:

```python
from etna.metrics import ClassificationMetrics, RegressionMetrics

# Classification evaluation
classifier_metrics = ClassificationMetrics()
accuracy = classifier_metrics.accuracy(y_true, y_pred)
precision, recall, f1 = classifier_metrics.precision_recall_f1(y_true, y_pred)

# Regression evaluation  
regression_metrics = RegressionMetrics()
mse = regression_metrics.mean_squared_error(y_true, y_pred)
r2 = regression_metrics.r2_score(y_true, y_pred)
```

### Preprocessing Module
Future integration points for data preprocessing:

```python
from etna.preprocessing import StandardScaler, OneHotEncoder

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode categorical variables
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical)
```

## Error Handling

The CLI implements comprehensive error handling:

1. **Build Errors**: Captures Cargo and Maturin output, provides actionable error messages
2. **Data Errors**: Validates JSON format, checks data shapes and types
3. **Model Errors**: Handles training failures, prediction errors
4. **File Errors**: Checks file existence, permissions, format validation

## Cross-Platform Support

### Windows
- PowerShell scripts (`etna.ps1`, `manage.ps1`)
- Batch file (`etna.bat`)
- Handles Windows path separators and command syntax

### Linux/macOS
- Bash-compatible command structure
- Unix path handling
- Standard shell integration

## Development Workflow

### Typical Development Cycle
1. **Setup**: `etna install --dev`
2. **Build**: `etna build`
3. **Test**: `etna test --coverage`
4. **Train**: `etna train --data examples/sample_data.json`
5. **Evaluate**: `etna metrics --pred predictions.json --true ground_truth.json`

### VS Code Integration
The project includes VS Code tasks for common operations:
- Build project (Ctrl+Shift+P → "Tasks: Run Task" → "ETNA: Build Project")
- Run tests
- Train sample model
- Clean build artifacts

### Continuous Integration
The CLI structure supports CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Build ETNA
  run: python etna_cli.py build --release

- name: Run Tests
  run: python etna_cli.py test --coverage

- name: Validate Training
  run: python etna_cli.py train --data examples/sample_data.json --epochs 10
```

## Extension Points

### Adding New Commands
1. Create a new argument parser in `create_argument_parser()`
2. Implement the command logic in `main()`
3. Add corresponding helper class if needed
4. Update help text and documentation

### Adding New Metrics
1. Extend `ETNAMetricsEvaluator` class
2. Add new evaluation methods
3. Update command-line arguments
4. Integrate with the metrics module

### Adding New Data Formats
1. Extend data loading methods in `ETNAModelManager`
2. Add format validation
3. Update documentation and examples

## Performance Considerations

### Build Optimization
- Release builds use `--release` flag for Rust optimization
- Maturin builds are cached between runs
- Incremental compilation reduces build times

### Runtime Performance
- Rust core provides optimal training/prediction performance
- Minimal Python overhead for data conversion
- Efficient memory management through Rust

### Scalability
- CLI supports batch processing of large datasets
- Configurable batch sizes and memory management
- Progress reporting for long-running operations

## Troubleshooting

### Common Issues

1. **"Cargo not found"**
   - Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

2. **"Maturin not found"**
   - Install: `pip install maturin`

3. **Build failures**
   - Check Rust version: `rustc --version`
   - Update dependencies: `cargo update`

4. **Import errors**
   - Rebuild: `python etna_cli.py clean && python etna_cli.py build`

5. **Permission errors on Windows**
   - Run PowerShell as Administrator
   - Check execution policy: `Set-ExecutionPolicy RemoteSigned`

### Debug Mode
Enable verbose output with environment variables:
```bash
export ETNA_DEBUG=1
python etna_cli.py build
```

## Future Enhancements

### Planned Features
1. **Model Serialization**: Save/load trained models
2. **Hyperparameter Optimization**: Grid search and random search
3. **Distributed Training**: Multi-GPU and multi-node support
4. **Model Visualization**: Training curves and network graphs
5. **Data Pipeline**: Advanced preprocessing and augmentation
6. **Web Interface**: Browser-based model training and evaluation

### API Extensions
1. **REST API**: HTTP endpoints for training and prediction
2. **gRPC Service**: High-performance model serving
3. **Docker Support**: Containerized deployment
4. **Cloud Integration**: AWS, GCP, Azure support

This comprehensive CLI implementation provides a solid foundation for the ETNA neural network framework, with excellent extensibility and cross-platform support.
