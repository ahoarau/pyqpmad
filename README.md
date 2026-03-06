# pyqpmad

A fast Python wrapper for [qpmad](https://github.com/asherikov/qpmad), a C++ Quadratic Programming (QP) solver based on Goldfarb-Idnani's active-set method. 
`pyqpmad` is built using [nanobind](https://github.com/wjakob/nanobind) for high-performance Python bindings and [Eigen](https://eigen.tuxfamily.org/) for efficient linear algebra.

## Features

- Solves Unconstrained, Constrained, and Bounded Quadratic Programs.
- Lightweight and fast wrapper utilizing `nanobind`.
- Automatic compilation of dependencies (`qpmad`, `eigen`, `nanobind`) from source via CMake.

## Examples

### Comprehensive Example

Here is a complete example demonstrating how to configure solver parameters and solve a QP problem with simple bounds and general linear constraints, utilizing the full API.

```python
import pyqpmad
import numpy as np

solver = pyqpmad.Solver()

# Problem definition (2 variables)
primal = np.zeros(2)

# Define the Hessian matrix (H) and linear term (h)
H = np.array([[2.0, 0.0], 
              [0.0, 2.0]], order='F')
h = np.array([-2.0, -2.0]) # Unconstrained minimum at [1.0, 1.0]

# Define simple bounds: 0.0 <= x_i <= 0.8
lb = np.array([0.0, 0.0])
ub = np.array([0.8, 0.8])

# Define the linear constraint: 1.0 <= x_0 + x_1 <= 1.5
A = np.array([[1.0, 1.0]], order='F')
Alb = np.array([1.0])
Aub = np.array([1.5])

# Configure solver parameters
params = pyqpmad.SolverParameters()
params.tolerance = 1e-8
params.max_iter = 100

# Solve with all constraints and parameters
status = solver.solve(primal, H, h, lb=lb, ub=ub, A=A, Alb=Alb, Aub=Aub, params=params)

if status == pyqpmad.ReturnStatus.OK:
    print(f"Optimal solution found: {primal}")
    print(f"Iterations: {solver.get_num_iterations()}")
    
    # Get active inequality duals
    duals = solver.get_inequality_dual()
    print(f"Active dual values: {duals.dual}")
    print(f"Active constraint indices: {duals.indices}")
else:
    print(f"Failed to solve. Status: {status}")
```

### Control Loop Example (Zero Allocation & Hessian Reuse)

In high-frequency control loops (e.g., Model Predictive Control), you can pre-allocate the solver's internal workspace using `reserve()` and reuse arrays to avoid memory allocations. If the Hessian matrix $H$ is constant, you can also avoid re-factorizing it at every step by keeping its inverted Cholesky factor.

```python
import pyqpmad
import numpy as np

num_variables = 2
num_simple_bounds = 2 # Length of lb/ub (must be 0 or num_variables)
num_general_constraints = 1 # Number of rows in A

solver = pyqpmad.Solver()
# Reserve memory: (primal_size, num_simple_bounds, num_general_constraints)
solver.reserve(num_variables, num_simple_bounds, num_general_constraints)

# Pre-allocate output array
primal = np.zeros(num_variables)

# Define constant terms
H = np.array([[2.0, 0.0], [0.0, 2.0]], order='F')
lb = np.array([-1.0, -1.0])
ub = np.array([1.0, 1.0])
A = np.array([[1.0, 1.0]], order='F')
Alb = np.array([-1.5])
Aub = np.array([1.5])

# Ask the solver to factorize H and return the inverted Cholesky factor 
# (this overwrites H with the factorization)
params = pyqpmad.SolverParameters()
params.return_inverted_cholesky_factor = True

# First solve to compute and store the factorization
h_initial = np.array([0.0, 0.0])
status = solver.solve(primal, H, h_initial, lb=lb, ub=ub, A=A, Alb=Alb, Aub=Aub, params=params)

# For subsequent solves, we tell the solver that H already contains the inverted Cholesky factor
params.hessian_type = pyqpmad.HessianType.HESSIAN_INVERTED_CHOLESKY_FACTOR
params.return_inverted_cholesky_factor = False # No need to factorize again

# Control loop (e.g., 1000 iterations)
h = np.zeros(num_variables)
for i in range(1000):
    # Only the linear term h (e.g., tracking error) changes in this control loop
    h[0] = np.sin(i * 0.01)
    h[1] = np.cos(i * 0.01)
    
    # Solve without memory allocation and without re-factorizing H
    status = solver.solve(primal, H, h, lb=lb, ub=ub, A=A, Alb=Alb, Aub=Aub, params=params)
    
    if status == pyqpmad.ReturnStatus.OK:
        # Apply optimal control action
        action = primal
    else:
        # Handle failure
        pass
```

## Installation

You can install the package directly from PyPI using `pip`:

```bash
pip install qpmad
```

### Building from Source

If you want to build the package from source, you can use `uv` or `pip`. 

#### Using `uv` (Recommended)

You can easily build and install this package using [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver.

To install it directly:
```bash
uv pip install .
```

To run the tests:
```bash
uv run --with pytest test_qpmad.py
```

#### Using `pip`

```bash
pip install .
```

To run tests with pip:
```bash
pip install pytest
pytest test_qpmad.py
```

## Build Explanation

The build process is managed entirely by CMake. When you install the package from source (via `pip` or `uv`), CMake's `FetchContent` module is triggered to automatically download the required dependencies:
- **`nanobind`**
- **`eigen`**
- **`qpmad`**

CMake then configures `qpmad` as an interface library linked with Eigen, and compiles the Python wrapper (`qpmad_pywrap.cpp`) using `nanobind`. Finally, Python type stubs (`.pyi`) are generated for IDE type hinting.

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