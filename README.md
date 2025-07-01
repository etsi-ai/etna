# etsi.etna

> **What if machine learning felt effortless?**

`etsi.etna` is a minimalistic, dependency-light neural network library
designed to make training and evaluating models on structured data fast,
interpretable, and beginner-friendly. It focuses on auto-preprocessing,
simple linear networks, and core metrics --- ideal for research
prototyping, learning, and quick deployments.

------------------------------------------------------------------------
## Installation

[Installation guide](https://github.com/etsi-ai/etna/blob/main/INSTALL.md)

------------------------------------------------------------------------

## 🚀 Quickstart

```bash
pip install etsi-etna
```

```python
import etsi.etna as etna

model = etna.Model("diabetes.csv", target="Outcome") 
model.train()
model.evaluate()
```

------------------------------------------------------------------------

## 🔮 Features

📦 One-liner dataset ingestion (.csv, .txt)

🧼 Automatic preprocessing (scaling, encoding)

🧠 Core NN: Linear → ReLU → Softmax

📊 Built-in evaluation (accuracy, F1)

🔍 CLI support

🪶 No hard dependencies --- minimal & fast

------------------------------------------------------------------------

## 🛠️ CLI Usage

ETNA provides a comprehensive command-line interface for building, training, and evaluating neural network models.

### Installation and Setup

1. **Install dependencies:**
```bash
# Install Rust (required for core implementation)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python dependencies
pip install maturin numpy pytest
```

2. **Build the project:**
```bash
# Build both Rust core and Python package
python etna_cli.py build

# Or use the convenience scripts:
# On Windows PowerShell:
.\etna.ps1 build

# On Windows Command Prompt:
etna.bat build
```

### Available Commands

#### Build Commands
```bash
# Build the entire project (Rust + Python)
python etna_cli.py build

# Build in release mode (optimized)
python etna_cli.py build --release

# Build only the Rust core
python etna_cli.py build --rust-only

# Build only the Python package
python etna_cli.py build --python-only
```

#### Testing Commands
```bash
# Run all tests (Rust + Python + Integration)
python etna_cli.py test

# Run only Rust tests
python etna_cli.py test --rust

# Run only Python tests
python etna_cli.py test --python

# Run tests with coverage report
python etna_cli.py test --coverage

# Run only integration tests
python etna_cli.py test --integration
```

#### Model Training
```bash
# Train a model with default settings
python etna_cli.py train --data examples/sample_data.json

# Train with custom configuration
python etna_cli.py train --data mydata.json --config config.json

# Train with specific parameters
python etna_cli.py train --data mydata.json --epochs 200 --lr 0.001
```

#### Making Predictions
```bash
# Make predictions on new data
python etna_cli.py predict --data test_data.json

# Use a specific model file
python etna_cli.py predict --data test_data.json --model my_model.pkl

# Save predictions to a specific file
python etna_cli.py predict --data test_data.json --output my_predictions.json
```

#### Model Evaluation
```bash
# Evaluate classification model
python etna_cli.py metrics --pred predictions.json --true ground_truth.json --task classification

# Evaluate regression model
python etna_cli.py metrics --pred predictions.json --true ground_truth.json --task regression
```

#### Project Management
```bash
# Create a new ETNA project
python etna_cli.py create my_new_project

# Create an advanced project template
python etna_cli.py create my_project --type advanced

# Clean build artifacts
python etna_cli.py clean

# Install dependencies
python etna_cli.py install

# Install development dependencies
python etna_cli.py install --dev
```

### Data Format

The CLI expects data in JSON format. Here's an example structure:

```json
{
  "X": [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
  ],
  "y": [0, 1, 1, 0]
}
```

For predictions, you can provide just the features:
```json
{
  "X": [
    [0.5, 0.5],
    [0.2, 0.8]
  ]
}
```

### Configuration File

Training configuration can be specified in a JSON file:

```json
{
  "epochs": 150,
  "learning_rate": 0.01,
  "batch_size": 32,
  "hidden_dim": 64,
  "optimizer": "sgd"
}
```

### Example Workflow

1. **Create a new project:**
```bash
python etna_cli.py create my_ml_project
cd my_ml_project
```

2. **Build the project:**
```bash
python etna_cli.py build
```

3. **Train a model:**
```bash
python etna_cli.py train --data examples/sample_data.json --epochs 100
```

4. **Make predictions:**
```bash
python etna_cli.py predict --data test_data.json
```

5. **Evaluate the model:**
```bash
python etna_cli.py metrics --pred predictions.json --true ground_truth.json
```

### Windows-Specific Usage

For Windows users, we provide convenient wrapper scripts:

**PowerShell:**
```powershell
.\etna.ps1 build
.\etna.ps1 train --data examples/sample_data.json
.\etna.ps1 test --coverage
```

**Command Prompt:**
```cmd
etna.bat build
etna.bat train --data examples/sample_data.json
etna.bat test --coverage
```

### Troubleshooting

**Common Issues:**

1. **Rust not found:** Install Rust from [rustup.rs](https://rustup.rs/)

2. **Maturin not found:** Install with `pip install maturin`

3. **Build failures:** Make sure you have the latest version of Rust and Python

4. **Permission errors on Windows:** Run PowerShell as Administrator

5. **Import errors:** Make sure the project is built with `python etna_cli.py build`

**Getting Help:**
```bash
python etna_cli.py --help
python etna_cli.py <command> --help
```

------------------------------------------------------------------------

## 🧠 Architecture

ETNA combines the performance of Rust with the convenience of Python:

- **Rust Core (`etna_core/`)**: High-performance neural network implementation
  - Linear layers with weight initialization
  - Activation functions (ReLU, Softmax)
  - Loss functions (Cross-entropy, MSE)
  - Optimizers (SGD, Adam)
  - PyO3 bindings for Python integration

- **Python Interface (`etna/`)**: User-friendly API and utilities
  - Model training and prediction API
  - Comprehensive evaluation metrics
  - Data preprocessing utilities
  - Command-line interface
  - Testing framework

------------------------------------------------------------------------

## 🤝 Contributing

Pull requests are welcome!

Please refer to [CONTRIBUTING.md](https://github.com/etsi-ai/etna/blob/main/CONTRIBUTING.md) for contribution guidelines and ensure
your code passes:

```bash
 make check 
```


Follow consistent commit messages 
Example: \[fix\] model.py: fixed one-hot bug

------------------------------------------------------------------------


## 📄 License

This project is distributed under the BSD-2-Clause License.
By contributing, you agree to license your code under the BSD-2-Clause license.


------------------------------------------------------------------------


Built with ❤️ by etsi.ai "Making machine learning simple, again."


------------------------------------------------------------------------

