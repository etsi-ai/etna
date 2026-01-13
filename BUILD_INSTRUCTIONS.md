# Building the Rust Extension for ETNA

This guide explains how to build the Rust extension for the ETNA project.

## Prerequisites

1. **Python 3.8+** - Already installed
2. **Rust toolchain** - Install from https://rustup.rs/ or it will be auto-installed during build
3. **Build tools** (Windows):
   - Microsoft C++ Build Tools (for compiling Rust on Windows)
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Quick Build (Recommended)

### Step 1: Install Build Dependencies

```powershell
# Install maturin and other dependencies
pip install maturin numpy pandas mlflow
```

### Step 2: Build and Install the Extension

From the project root directory (`C:\Github\etna`):

```powershell
# For development (faster builds, includes debug info)
maturin develop

# OR for release (optimized, slower build)
maturin develop --release
```

This will:
- Compile the Rust code in `etna_core/`
- Build the Python extension module
- Install it in your current Python environment

## Alternative: Using pip (Editable Install)

You can also use pip, which will automatically use maturin:

```powershell
# Install in editable/development mode
pip install -e .

# This is equivalent to running maturin develop
```

## Verify Installation

After building, verify the extension is working:

```python
import etna
from etna import Model

# This should not raise an ImportError
print("✅ Rust extension loaded successfully!")
```

## Troubleshooting

### Issue: "Rust not found" or "Cargo not found"

**Solution**: Install Rust manually:
1. Download and run `rustup-init.exe` from https://rustup.rs/
2. Follow the installation wizard
3. Restart your terminal/PowerShell
4. Verify: `rustc --version` and `cargo --version`

### Issue: "Microsoft C++ Build Tools not found" (Windows)

**Solution**: Install Microsoft C++ Build Tools:
1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. During installation, select "Desktop development with C++"
3. Restart your terminal after installation

### Issue: Build fails with linker errors

**Solution**: 
- Make sure you have the latest Rust toolchain: `rustup update`
- On Windows, ensure Visual Studio Build Tools are installed
- Try cleaning and rebuilding: `maturin develop --release`

### Issue: "maturin: command not found"

**Solution**: Install maturin:
```powershell
pip install maturin
```

## Build Options

### Development Build (Fast, with debug info)
```powershell
maturin develop
```

### Release Build (Optimized, slower)
```powershell
maturin develop --release
```

### Build Only (Don't install)
```powershell
maturin build
```

### Build Wheel (For distribution)
```powershell
maturin build --release
```

## Project Structure

```
etna/
├── etna/              # Python package
│   ├── api.py         # Main API (uses _etna_rust)
│   └── ...
├── etna_core/         # Rust crate
│   ├── Cargo.toml     # Rust dependencies
│   └── src/
│       ├── lib.rs     # Python bindings (pyo3)
│       └── ...
└── pyproject.toml      # Maturin build config
```

The `pyproject.toml` tells maturin:
- Module name: `etna._etna_rust`
- Rust manifest: `etna_core/Cargo.toml`
- Python source: current directory

## After Building

Once built, you can:
1. Run tests: `python test_validation_split.py`
2. Use the API: `from etna import Model`
3. Check MLflow integration (if MLflow server is running)

## Notes

- The first build will take longer (downloads Rust dependencies)
- Subsequent builds are faster (incremental compilation)
- Use `--release` for production/performance testing
- Use regular `develop` for faster iteration during development

