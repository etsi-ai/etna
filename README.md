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
* **Arbitrary Depth**: Etna now supports sequential multi-layer architectures.
* **Stateful Optimization**: Advanced optimizers like Adam maintain momentum and moment estimates across training sessions.

---

## Key Features

* **Sequential Multi-Layer Architecture**: Define any number of hidden layers using a simple list (e.g., `[64, 32, 16]`).
* **Hybrid Architecture**: `pyo3` bindings bridge Python ease with Rust performance.
* **Auto-Preprocessing**: Automatic scaling and categorical encoding based on column types.
* **Persistent Training**: Save and load models while preserving weight values and optimizer states.
* **Flexible MLflow Tracking**: Single-line experiment tracking that is now safely optional for local-only use.

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
    pip install -r requirements.txt

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

1. **Classification with Deep Architecture**
Etna now supports any network depth. The example below initializes a 4-layer network.
```bash
from etna import Model

# Initialize model with 3 hidden layers: 64 -> 32 -> 16
model = Model(
    file_path="iris.csv",
    target="species",
    hidden_layers=[64, 32, 16],
    activation="leaky_relu"
)

# Train with Adam optimizer
model.train(epochs=100, lr=0.01, optimizer="adam")

# Predict original class labels
predictions = model.predict()
```

2. **Regression & Incremental Training**
Because optimizer states are persistent, you can resume training smoothly.
```bash
model = Model("housing.csv", target="price", task_type="regression")

# First training phase
model.train(epochs=50)

# Resume training: Adam momentum is preserved!
model.train(epochs=50)
```

---

## Experiment Tracking

Etna includes native MLflow integration. To use it, simply provide your tracking URI when saving.
```bash
# Save locally AND log to MLflow in one step
model.save_model(
    path="my_model_v1.json",
    run_name="Deep_Run_01",
    mlflow_tracking_uri="http://localhost:5000"
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
