@echo off
REM ETNA CLI Batch Script
REM This script provides a convenient way to run ETNA commands on Windows Command Prompt

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Set the Python path to include the project directory
set PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%

REM Execute the Python CLI with all arguments
python "%SCRIPT_DIR%etna_cli.py" %*
