# Installation Guide for ETNA

## AUR:

We have packaged `etna-git` for the AUR. For now, `etna-bin` is not available, but it may be packaged by the community in the future.

## Building:

ETNA installs to `/usr/local/bin/` by default. You can change this behavior by editing the [Makefile](../Makefile) variable, `DESTDIR`, which acts as a prefix for all installed files. Alternatively, specify it in the make command line. For example, to install everything in a specific directory, run:



## Dependencies:

**Runtime:**
-   Python 3
-   PyTorch
-   Pandas
-   Numpy

**Compile time:**
-   git
-   make

## Compiling:

1. Clone the repository:
    ```
    git clone https://github.com/etsi-ai/etna.git
    cd etna
    ```

2. Setup dependencies and environment:
    ```
    make setup
    ```

3. Build the project:
    ```
    make clean
    make
    ```

4. Install the project:
    ```
    sudo make install
    ```

## Running:

Once installed, you can start using the ETNA model. Here’s a basic usage example:

```python
import etsi.etna as etna

model = etna.Model("diabetes.csv", target="Outcome")
model.train()
model.evaluate()


### Explanation:
- **AUR** section: Explains the AUR package and installation steps.
- **Building**: Informs about default installation paths and how to change the destination directory for installation.
- **Dependencies**: Lists runtime and compile-time dependencies required to build and run ETNA.
- **Compiling**: Provides a step-by-step guide to clone, build, and install ETNA.
- **Running**: Gives an example of how to use the package after installation.

This is tailored to be clear and easy to follow for anyone looking to install and run your ETNA library.
