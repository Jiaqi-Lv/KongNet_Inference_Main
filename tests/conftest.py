"""
Pytest configuration for KongNet Inference tests.

This file configures pytest to handle missing dependencies gracefully
and provides custom markers for different test categories.
"""

import sys

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "requires_torch: marks tests that require PyTorch"
    )
    config.addinivalue_line("markers", "requires_numpy: marks tests that require NumPy")
    config.addinivalue_line(
        "markers", "requires_tiatoolbox: marks tests that require TIAToolbox"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")


def pytest_runtest_setup(item):
    """Setup function to skip tests based on missing dependencies"""

    # Skip tests requiring torch if not available
    if item.get_closest_marker("requires_torch"):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

    # Skip tests requiring numpy if not available
    if item.get_closest_marker("requires_numpy"):
        try:
            import numpy
        except ImportError:
            pytest.skip("NumPy not available")

    # Skip tests requiring tiatoolbox if not available
    if item.get_closest_marker("requires_tiatoolbox"):
        try:
            import tiatoolbox
        except ImportError:
            pytest.skip("TIAToolbox not available")


def pytest_collection_modifyitems(config, items):
    """Modify collected test items"""

    # Add markers based on test names/paths
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if "slow" in item.nodeid or "end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.slow)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Fixture to provide a temporary cache directory"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_wsi_path(tmp_path):
    """Fixture to provide a mock WSI file path"""
    wsi_file = tmp_path / "test_sample.svs"
    wsi_file.write_text("mock wsi file")
    return str(wsi_file)


@pytest.fixture
def sample_detection_records():
    """Fixture providing sample detection records for testing"""
    return [
        {"x": 100, "y": 200, "type": "mitotic_figure", "prob": 0.95},
        {"x": 300, "y": 400, "type": "mitotic_figure", "prob": 0.87},
        {"x": 500, "y": 600, "type": "lymphocyte", "prob": 0.92},
    ]


@pytest.fixture
def mock_model_config():
    """Fixture providing mock model configuration"""
    return {"num_heads": 1, "decoders_out_channels": [3]}
