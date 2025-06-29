#!/usr/bin/env python3
"""
ETNA CLI Entry Point

This script provides a command-line entry point for the ETNA neural network framework.
It can be used directly or installed as a console script.

Usage:
    python etna_cli.py build
    python etna_cli.py test
    python etna_cli.py train --data examples/sample_data.json
"""

import sys
import os

# Add the etna package to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from etna.cli import main

if __name__ == '__main__':
    sys.exit(main())
