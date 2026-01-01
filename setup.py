"""
Python package setup for ETNA neural network framework.
"""

from setuptools import setup, find_packages

setup(
    name="etna",
    version="0.1.0",
    description="A neural network framework with Rust core and Python interface",
    long_description=open("README.md").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    author="ETNA Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.0.0",
        "mlflow>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="neural-network machine-learning rust python deep-learning",
    entry_points={
        'console_scripts': [
            'etna=etna.cli:main',
        ],
    },
)