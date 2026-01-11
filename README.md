# etsi.etna
### High-Performance Neural Networks. Rust Core. Python Ease.

<div align="center">

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-black)](https://www.rust-lang.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Integrated-blueviolet)](https://mlflow.org/)

</div>

---

## 🏗️ Technical Architecture

Etna operates as a hybrid system where the high-level orchestration is managed by Python, while the performance-critical computation is offloaded to a compiled Rust core.

### 1. The PyO3 Bridge
The following diagram illustrates the FFI (Foreign Function Interface) boundary. Python's `api.py` acts as a wrapper that invokes native Rust methods through PyO3 bindings.

```mermaid
graph TD
    subgraph Python_Frontend ["Python API (api.py)"]
        A[Model Class] -->|Method Call| B[PyO3 Bound Method]
        B -->|Return PyObject| A
    end

    subgraph Bridge_Layer ["PyO3 / FFI Layer"]
        B -->|Data Marshaling| C[Native Rust Bindings]
        C -->|Unbox/Convert| D[Rust Primitive Types]
    end

    subgraph Rust_Engine ["Rust Core (etna_core)"]
        D --> E[Neural Network Engine]
        E --> F[Backprop & Optimization]
        F -->|Result Vectors| C
    end
```

### 2. Preprocessing Data Flow
Data follows a specific pipeline before reaching the training loop. Raw inputs are processed in Python using optimized Pandas/NumPy routines before being serialized into the Rust backend.
```mermaid
flowchart LR
    A[(Raw CSV)] -->|Pandas| B[Python Preprocessor]
    
    subgraph Python_Logic [Data Conditioning]
        B --> C{Categorical?}
        C -->|Yes| D[One-Hot Encoding]
        C -->|No| E[Standard Scaling]
    end

    D & E --> F[Float64 Buffer]
    F -->|Zero-copy/Maturin| G[Rust Tensor Core]

    subgraph Rust_Backend [Training Loop]
        G --> H[Input Layer]
        H --> I[Rust Forward Pass]
    end
```

### 3. Smart Task Detection Logic
Etna's `__init__` branching logic automatically determines the mathematical objective (Classification vs. Regression) based on target variable heuristics.
```mermaid
graph TD
    Start([Model.__init__]) --> Input{Target Data Type}
    
    Input -->|String/Object| Task_C[Task: Classification]
    Input -->|Numeric| Check_Card{Cardinality / Unique < 10%}
    
    Check_Card -->|True| Task_C
    Check_Card -->|False| Task_R[Task: Regression]
    
    Task_C --> Map_C[Loss: CrossEntropy / Map: String-to-Int]
    Task_R --> Map_R[Loss: MSE / Map: Float64]
    
    Map_C & Map_R --> Init_Rust[Initialize Rust Structs]
```

---

## 🔧 Technical Troubleshooting
Building and linking Rust extensions requires a specific toolchain configuration.
* **Maturin Build Failures:** Ensure `pip install maturin` is updated to the latest version. If build fails on linkers, verify that the `target` directory has write permissions.
* **C++ Compiler Requirement (Windows):** Even though the core is Rust, PyO3 requires the MSVC linker. Ensure "Desktop development with C++" is installed via Visual Studio Installer.
* **Architecture Mismatches:** If using macOS Silicon, ensure both Python and Rust are targeting `aarch64-apple-darwin`.

---

## 🚀 Development Setup
**Build from Source**
```
# Clone and enter repo
git clone [https://github.com/etsi-ai/etna.git](https://github.com/etsi-ai/etna.git) && cd etna

# Build Rust extensions and install in dev mode
maturin develop --release
```
**Internal Metrics Tracking**
Etna utilizes MLflow to track Rust-calculated gradients and loss values. These are passed back across the bridge at the end of each epoch to minimize FFI overhead.

---

## 📄 License

This project is distributed under the **BSD-2-Clause License**. See the [LICENSE](LICENSE) file for complete details.

---

<div align="center">

**Built with ❤️ by the etsi.ai Team**

[⬆ Back to Top](#etsietna)

</div>
