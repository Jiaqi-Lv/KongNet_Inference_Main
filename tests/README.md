# KongNet Inference Tests

This directory contains the test suite for the KongNet Inference Main project.

## Test Structure

- `test_inference_pipelines.py` - Tests for the core inference pipeline classes (MIDOG, PanNuke, CoNIC, MONKEY)
- `test_interface.py` - Tests for the command-line interface and argument handling
- `test_data_utils.py` - Tests for data processing utilities and helper functions
- `test_model.py` - Tests for the KongNet model architecture
- `conftest.py` - Pytest configuration and shared fixtures
- `run_tests.py` - Test runner script with dependency checking

## Running Tests

### Option 1: Using pytest

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
pytest

# Run specific test file
pytest tests/test_inference_pipelines.py

# Run tests with verbose output
pytest -v
```

### Option 2: Using python

```bash
python tests/run_tests.py
```



### Required for Basic Tests
- Python 3.10+
- unittest (built-in)

### Optional Dependencies
- `pytest` - For advanced test features
- `pytest-mock` - For enhanced mocking capabilities

## Example Test Run

```bash
$ python tests/run_tests.py

Test Environment Dependencies:
------------------------------
✓ torch
✓ numpy
✗ tiatoolbox
✓ zarr
✓ skimage
------------------------------
Missing dependencies: tiatoolbox
Some tests may be skipped.

Running KongNet Inference Test Suite
==================================================
test_initialization (test_inference_pipelines.TestMIDOGInference) ... ok
test_model_config (test_inference_pipelines.TestMIDOGInference) ... ok
test_target_channels (test_inference_pipelines.TestMIDOGInference) ... ok
...
==================================================
Test Summary:
Tests run: 45
Failures: 0
Errors: 0
Skipped: 3
```