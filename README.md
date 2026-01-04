<div align="center">

# etsi.etna
### High-Performance Neural Networks. Rust Core. Python Ease.

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-black)](https://www.rust-lang.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Integrated-blueviolet)](https://mlflow.org/)

> **What if machine learning felt effortless?**

`etsi.etna` is a minimalistic, dependency-light neural network library
designed to make training and evaluating models on structured data fast,
interpretable and beginner-friendly. It focuses on auto-preprocessing,
simple linear networks, and core metrics - ideal for research
prototyping, learning, and quick deployments.

[Features](#key-features) • [Installation](#installation) • [MVP Demo](#run-the-mvp-demo) • [Quickstart](#quickstart) • [Experiment Tracking](#experiment-tracking)

</div>

---

## Why Etna?

Machine learning libraries often force a trade-off: simplicity or speed. Etna removes that barrier.

* **Blazing Fast**: The heavy lifting (Linear layers, ReLU, Softmax, Backprop) is handled by a highly optimized **Rust** core (`etna_core`).
* **Pythonic API**: Users interact with a familiar, Scikit-learn-like Python interface.
* **Intelligent Defaults**: Automatically detects if you are performing **Classification** or **Regression** based on your target data.
* **Production Ready**: Built-in **MLflow** integration for experiment tracking and model versioning out of the box.

---

## Key Features

* **Hybrid Architecture**: `pyo3` bindings bridge Python ease with Rust performance.
* **Auto-Preprocessing**: Automatic scaling (Standard/MinMax) and categorical encoding (One-Hot) based on column types.
* **Smart Task Detection**:
    * *Classification*: Auto-detects low cardinality or string targets.
    * *Regression*: Auto-detects continuous numeric targets.
* **Comprehensive Metrics**: Built-in evaluation suite including Accuracy, F1-Score, MSE, R², and Cross-Entropy Loss.
* **Zero-Config MLflow**: Save, version, and track model metrics with a single line of code.

---

## Installation

### Prerequisites
* Python (3.8 or later)
* Rust (1.70 or later)

### From Source (Development)
Etna uses `maturin` to build the Rust extensions.

1.  **Clone the repository**
    ```bash
    git clone https://github.com/etsi-ai/etna.git
    cd etna
    ```
    
2.  **Set up a Virtual Environment (Recommended)**
    ```bash
    python -m venv .venv

    # Activate the environment
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate   # Windows
    ```

3.  **Install dependencies & build**
    ```bash
    # Install build tools
    pip install maturin numpy pandas mlflow jupyter pytest
    
    # Build and install locally
    maturin develop --release
    ```

---

## Run the MVP Demo

The best way to see Etna in action is to run our interactive MVP notebook. This notebook verifies your installation by performing an end-to-end test of the entire system.

It will automatically:
1.  **Generate Dummy Data**: Creates synthetic datasets for both classification and regression.
2.  **Train Models**: Trains the Rust backend on both tasks.
3.  **Track Experiments**: Logs loss curves and artifacts to a local MLflow server.

To run it:
```bash
jupyter notebook mvp_testing.ipynb
```

---

## Quickstart

If you prefer to start coding immediately, here are the basics:

1. **Classification (Auto-Detected)**
Etna automatically handles string labels and normalizes your data.
```bash
from etna import Model
from etna.metrics import accuracy_score

# Initialize model (Auto-detects Classification based on target)
model = Model(file_path="iris.csv", target="species")

# Train with Rust backend
model.train(epochs=100, lr=0.01)

# Predict (Returns original class labels, e.g., "setosa")
predictions = model.predict()
print("Predictions:", predictions[:5])
```

2. **Regression (Manual Override)**
You can explicitly define the task type if needed.
```bash
# Force regression for continuous targets
model = Model(file_path="housing.csv", target="price", task_type="regression")

model.train(epochs=500, lr=0.001)

# Predict (Returns float values)
prices = model.predict()
```

---

## Experiment Tracking

Etna includes native MLflow integration. Track your loss curves, parameters, and artifacts without setting up complex boilerplate.
```bash
# Train your model
model.train(epochs=200)

# Save locally AND log to MLflow in one step
model.save_model(
    path="my_model_v1.json", 
    run_name="MVP_Demo_Run"
)
```
**What happens automatically:**

* Model artifact saved to `my_model_v1.json`
* Parameters (`task_type`, `target`) logged to MLflow
* Training Loss history logged as metrics
* Artifacts uploaded to the MLflow run

View your dashboard by running `mlflow ui` in your terminal and visiting `http://localhost:5000`

---

## Contributing

Pull requests are welcome!

Please refer to [CONTRIBUTING.md](https://github.com/etsi-ai/etna/blob/main/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](https://github.com/etsi-ai/etna/blob/main/CODE_OF_CONDUCT.md) before submitting a Pull Request.

---

## Join the Community

Connect with the **etsi.ai** team and other contributors on our Discord.

[![Discord](https://img.shields.io/badge/Discord-Join%20the%20Server-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/invite/VCeY6H72rq)

---

## License

This project is distributed under the **BSD-2-Clause License**. See the [LICENSE](https://github.com/etsi-ai/etna/blob/main/LICENSE) for details.

---

> Built with ❤️ by etsi.ai
