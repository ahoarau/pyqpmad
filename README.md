# pyqpmad

A fast Python wrapper for [qpmad](https://github.com/asherikov/qpmad), a C++ Quadratic Programming (QP) solver based on Goldfarb-Idnani's active-set method. 
`pyqpmad` is built using [nanobind](https://github.com/wjakob/nanobind) for high-performance Python bindings and [Eigen](https://eigen.tuxfamily.org/) for efficient linear algebra.

## Features

- Solves Unconstrained, Constrained, and Bounded Quadratic Programs.
- Lightweight and fast wrapper utilizing `nanobind`.
- Automatic compilation of dependencies (`qpmad`, `eigen`, `nanobind`) from source via CMake.

## Installation

### Using `uv` (Recommended)

You can easily build and install this package using [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver.

To install it directly:
```bash
uv pip install .
```

To run the tests:
```bash
uv pip install pytest
uv run pytest test_qpmad.py
```

### Using `pip`

```bash
pip install .
```

To run tests with pip:
```bash
pip install pytest
pytest test_qpmad.py
```

## Build Explanation

The build process is managed entirely by CMake. When you install the package (via `pip` or `uv`), CMake's `FetchContent` module is triggered to automatically download the required dependencies:
- **`nanobind`** (v2.12.0)
- **`eigen`** (v5.0.1 mirror)
- **`qpmad`** (v1.4.0)

CMake then configures `qpmad` as an interface library linked with Eigen, and compiles the Python wrapper (`qpmad_pywrap.cpp`) using `nanobind`. Finally, Python type stubs (`.pyi`) are generated for IDE type hinting.

## Example Usage

Here is a quick example demonstrating how to solve a constrained QP problem.

**Problem:** Minimize $x^2 + y^2$ subject to $x + y = 1$

```python
import pyqpmad
import numpy as np

solver = pyqpmad.Solver()

# Define the Hessian matrix (H) and linear term (h)
H = np.array([[2.0, 0.0], 
              [0.0, 2.0]], order='F')
h = np.array([0.0, 0.0])

# Define the linear constraint: 1.0 <= x + y <= 1.0
A = np.array([[1.0, 1.0]], order='F')
Alb = np.array([1.0]) # Lower bound
Aub = np.array([1.0]) # Upper bound

# Pre-allocate output array
primal = np.zeros(2)

# Solve
status = solver.solve(primal, H.copy(order='F'), h, A=A, Alb=Alb, Aub=Aub)

if status == pyqpmad.ReturnStatus.OK:
    print(f"Optimal solution found: {primal}") # Output: [0.5 0.5]
else:
    print(f"Failed to solve. Status: {status}")
```

For more examples—such as solving unconstrained problems or adding lower/upper bounds—please check `test_qpmad.py`.

## Release Pipeline

This project uses `cibuildwheel` integrated via GitHub Actions to automatically build wheels for Linux, macOS, and Windows.

### Triggering a Release

To publish a new release to PyPI, tag the main branch with a version number starting with `v` (e.g., `v1.0.0`) and push the tag to GitHub:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The GitHub Actions workflow will:
1. Build `sdist` and wheels for all major platforms.
2. Run tests on the generated wheels.
3. Automatically publish the artifacts to PyPI.
