# ETNA Project Management Script
# This PowerShell script provides easy-to-use commands for managing the ETNA project

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("help", "setup", "build", "build-dev", "build-release", "test", "test-rust", "test-python", "clean", "install", "dev-install", "demo")]
    [string]$Action
)

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ETNA_CLI = "$ProjectDir\etna_cli.py"

function Show-Help {
    Write-Host "ETNA Project Management Script" -ForegroundColor Green
    Write-Host "==============================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Cyan
    Write-Host "  help        - Show this help message"
    Write-Host "  setup       - Initial project setup (install dependencies)"
    Write-Host "  build       - Build the project (Rust + Python)"
    Write-Host "  build-dev   - Build for development"
    Write-Host "  build-release - Build optimized release version"
    Write-Host "  test        - Run all tests"
    Write-Host "  test-rust   - Run only Rust tests"
    Write-Host "  test-python - Run only Python tests"
    Write-Host "  clean       - Clean build artifacts"
    Write-Host "  install     - Install the package"
    Write-Host "  dev-install - Install in development mode"
    Write-Host "  demo        - Run a quick demo"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\manage.ps1 setup"
    Write-Host "  .\manage.ps1 build"
    Write-Host "  .\manage.ps1 test"
    Write-Host "  .\manage.ps1 demo"
}

function Invoke-Setup {
    Write-Host "🔧 Setting up ETNA development environment..." -ForegroundColor Green
    
    # Check if Rust is installed
    try {
        $rustVersion = cargo --version
        Write-Host "✅ Rust found: $rustVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Rust not found. Please install from https://rustup.rs/" -ForegroundColor Red
        return
    }
    
    # Install Python dependencies
    Write-Host "📦 Installing Python dependencies..." -ForegroundColor Cyan
    pip install maturin numpy pytest pytest-cov black flake8
    
    Write-Host "✅ Setup complete!" -ForegroundColor Green
}

function Invoke-Build {
    param([string]$Mode = "default")
    
    Write-Host "🏗️ Building ETNA project..." -ForegroundColor Green
    
    $args = @("build")
    if ($Mode -eq "release") {
        $args += "--release"
    }
    
    & python $ETNA_CLI @args
}

function Invoke-Test {
    param([string]$TestType = "all")
    
    Write-Host "🧪 Running tests..." -ForegroundColor Green
    
    $args = @("test")
    switch ($TestType) {
        "rust" { $args += "--rust" }
        "python" { $args += "--python" }
        default { $args += "--coverage" }
    }
    
    & python $ETNA_CLI @args
}

function Invoke-Clean {
    Write-Host "🧹 Cleaning build artifacts..." -ForegroundColor Green
    & python $ETNA_CLI clean
}

function Invoke-Install {
    param([bool]$Dev = $false)
    
    if ($Dev) {
        Write-Host "📦 Installing ETNA in development mode..." -ForegroundColor Green
        & python $ETNA_CLI install --dev
    } else {
        Write-Host "📦 Installing ETNA..." -ForegroundColor Green
        & python $ETNA_CLI install
    }
}

function Invoke-Demo {
    Write-Host "🎯 Running ETNA demo..." -ForegroundColor Green
    
    # Build first
    Write-Host "Building project..." -ForegroundColor Cyan
    & python $ETNA_CLI build
    
    if ($LASTEXITCODE -eq 0) {
        # Train a model
        Write-Host "Training demo model..." -ForegroundColor Cyan
        & python $ETNA_CLI train --data examples/sample_data.json --epochs 50
        
        # Make predictions
        Write-Host "Making predictions..." -ForegroundColor Cyan
        & python $ETNA_CLI predict --data examples/test_data.json
        
        # Evaluate
        Write-Host "Evaluating model..." -ForegroundColor Cyan
        & python $ETNA_CLI metrics --pred predictions.json --true examples/ground_truth.json
        
        Write-Host "✅ Demo complete!" -ForegroundColor Green
    } else {
        Write-Host "❌ Build failed. Cannot run demo." -ForegroundColor Red
    }
}

# Main execution
switch ($Action) {
    "help" { Show-Help }
    "setup" { Invoke-Setup }
    "build" { Invoke-Build }
    "build-dev" { Invoke-Build -Mode "dev" }
    "build-release" { Invoke-Build -Mode "release" }
    "test" { Invoke-Test }
    "test-rust" { Invoke-Test -TestType "rust" }
    "test-python" { Invoke-Test -TestType "python" }
    "clean" { Invoke-Clean }
    "install" { Invoke-Install }
    "dev-install" { Invoke-Install -Dev $true }
    "demo" { Invoke-Demo }
    default { Show-Help }
}
