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

### Option 1: Using pytest (Recommended)

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
pytest

# Run specific test file
pytest tests/test_inference_pipelines.py

# Run tests with verbose output
pytest -v

# Run only fast tests (exclude slow/integration tests)
pytest -m "not slow"

# Run tests that don't require heavy dependencies
pytest -m "not requires_torch"
```

### Option 2: Using the custom test runner

```bash
# Run unit tests with dependency checking
python tests/run_tests.py

# Run tests in mock mode (without dependencies)
python tests/run_tests.py --mock

# Run integration tests (when implemented)
python tests/run_tests.py --integration
```

### Option 3: Using unittest directly

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_inference_pipelines

# Run specific test class
python -m unittest tests.test_inference_pipelines.TestMIDOGInference
```

## Test Categories

### Unit Tests
- **Pipeline Tests**: Test individual inference pipeline classes
- **Interface Tests**: Test command-line argument handling and workflow
- **Model Tests**: Test model architecture and configuration
- **Utility Tests**: Test data processing and helper functions

### Integration Tests (Future)
- End-to-end inference pipeline tests
- Model loading and inference tests
- File I/O and caching tests
- Performance benchmarking

## Dependencies

The tests are designed to handle missing dependencies gracefully:

### Required for Basic Tests
- Python 3.10+
- unittest (built-in)

### Optional Dependencies
- `pytest` - For advanced test features
- `pytest-mock` - For enhanced mocking capabilities
- `torch` - Required for model and tensor tests
- `numpy` - Required for array processing tests
- `tiatoolbox` - Required for WSI processing tests

### Test Dependencies
Tests will automatically skip if required dependencies are not available.

## Test Design Principles

1. **Graceful Degradation**: Tests skip when dependencies are missing rather than failing
2. **Mocking**: Heavy dependencies are mocked to test logic without requiring full installation
3. **Isolation**: Each test is independent and doesn't rely on external files or state
4. **Comprehensive Coverage**: Tests cover configuration, initialization, processing logic, and error handling
5. **Fast Execution**: Most tests run quickly; slow tests are marked appropriately

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

## Adding New Tests

When adding new functionality, please add corresponding tests:

1. **New Pipeline**: Add test class in `test_inference_pipelines.py`
2. **New Utility Function**: Add tests in `test_data_utils.py`
3. **New CLI Feature**: Add tests in `test_interface.py`
4. **New Model Component**: Add tests in `test_model.py`

### Test Template

```python
class TestNewFeature(unittest.TestCase):
    """Test description"""
    
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def tearDown(self):
        """Clean up after tests"""
        pass
    
    def test_feature_functionality(self):
        """Test specific functionality"""
        # Arrange
        # Act  
        # Assert
        pass
```

## Continuous Integration

These tests are designed to work in CI environments where dependencies may be limited. The graceful dependency handling ensures tests can run in various environments while providing maximum coverage when all dependencies are available.