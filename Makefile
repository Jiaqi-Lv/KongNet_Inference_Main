# Makefile for KongNet Inference Main

.PHONY: test test-unit test-mock test-verbose clean lint help

# Default target
help:
	@echo "Available targets:"
	@echo "  test        - Run all unit tests"
	@echo "  test-mock   - Run tests in mock mode (no dependencies required)"
	@echo "  test-verbose - Run tests with verbose output"
	@echo "  test-fast   - Run only fast tests (skip slow ones)"
	@echo "  test-coverage - Run tests with coverage report (requires coverage.py)"
	@echo "  lint        - Run code linting (requires flake8)"
	@echo "  clean       - Clean up cache and temporary files"
	@echo "  install-dev - Install development dependencies"

# Run unit tests
test:
	python tests/run_tests.py

# Run tests in mock mode (no heavy dependencies)
test-mock:
	python tests/run_tests.py --mock

# Run tests with verbose output
test-verbose:
	python -m unittest discover tests -v

# Run only fast tests using pytest
test-fast:
	pytest -m "not slow" -v

# Run tests with coverage
test-coverage:
	coverage run -m pytest
	coverage report -m
	coverage html

# Lint code
lint:
	flake8 inference/ model/ tests/ --max-line-length=100 --ignore=E203,W503

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage

# Install development dependencies
install-dev:
	pip install pytest pytest-mock coverage flake8

# Run inference examples (if test data available)
test-examples:
	@echo "Running example inference commands..."
	@if [ -d "test_input" ] && [ ! -z "$$(ls -A test_input)" ]; then \
		echo "Running MIDOG inference..."; \
		python inference_MIDOG.py --input_dir test_input --output_dir test_output --no_tta; \
	else \
		echo "No test input files found. Skipping example runs."; \
	fi

# Check test environment
check-env:
	python tests/run_tests.py --mock | grep -A 10 "Test Environment Dependencies"