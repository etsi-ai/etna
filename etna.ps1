# ETNA CLI PowerShell Script
# This script provides a convenient way to run ETNA commands on Windows

param(
    [Parameter(Mandatory=$true)]
    [string[]]$Command
)

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Set the Python path to include the project directory
$env:PYTHONPATH = "$ScriptDir;$env:PYTHONPATH"

# Build the command
$PythonArgs = @("$ScriptDir\etna_cli.py") + $Command

# Execute the command
& python @PythonArgs
