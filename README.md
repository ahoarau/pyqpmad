# pyqpmad
Python wrapper for qpmad

## Build and Test Locally

To build and install the package locally, run:

```bash
pip install .
```

To run the tests, install `pytest` and execute:

```bash
pip install pytest
pytest test_qpmad.py
```

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
