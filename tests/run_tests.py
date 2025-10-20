"""
Test runner and configuration for the KongNet Inference test suite.

This module provides utilities for running the complete test suite,
handling test dependencies, and generating test reports.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestEnvironment:
    """Manage test environment and dependencies"""

    @staticmethod
    def check_dependencies():
        """Check which dependencies are available for testing"""
        deps = {
            "torch": False,
            "numpy": False,
            "tiatoolbox": False,
            "zarr": False,
            "skimage": False,
        }

        try:
            import torch

            deps["torch"] = True
        except ImportError:
            pass

        try:
            import numpy

            deps["numpy"] = True
        except ImportError:
            pass

        try:
            import tiatoolbox

            deps["tiatoolbox"] = True
        except ImportError:
            pass

        try:
            import zarr

            deps["zarr"] = True
        except ImportError:
            pass

        try:
            import skimage

            deps["skimage"] = True
        except ImportError:
            pass

        return deps

    @staticmethod
    def print_dependency_status():
        """Print status of test dependencies"""
        deps = TestEnvironment.check_dependencies()
        print("Test Environment Dependencies:")
        print("-" * 30)
        for name, available in deps.items():
            status = "✓" if available else "✗"
            print(f"{status} {name}")
        print("-" * 30)

        missing = [name for name, available in deps.items() if not available]
        if missing:
            print(f"Missing dependencies: {', '.join(missing)}")
            print("Some tests may be skipped.")
        else:
            print("All dependencies available!")
        print()


def run_unit_tests():
    """Run the unit test suite"""

    # Print dependency status
    TestEnvironment.print_dependency_status()

    # Discover and run tests
    loader = unittest.TestLoader()

    # Load tests from the tests directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(test_dir, pattern="test_*.py")

    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2, stream=sys.stdout, descriptions=True, failfast=False
    )

    print("Running KongNet Inference Test Suite")
    print("=" * 50)

    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print(f"\nFailures ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")

    # Return True if all tests passed
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":

    success = True
    success &= run_unit_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
