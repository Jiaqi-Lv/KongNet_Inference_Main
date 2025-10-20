"""
Unit tests for the base WSI inference classes and pipelines.

Tests the abstract base class and concrete implementations for different datasets:
- MIDOG (mitotic figure detection)
- PanNuke (multi-class cell segmentation) 
- CoNIC (6-class cell classification)
- MONKEY (inflammatory cell detection)
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

from inference.wsi_inference_base import BaseWSIInference
from inference.wsi_inference_CoNIC import CoNICInference
from inference.wsi_inference_MIDOG import MIDOGInference
from inference.wsi_inference_MONKEY import MONKEYInference
from inference.wsi_inference_PanNuke import PanNukeInference


class TestBaseWSIInference(unittest.TestCase):
    """Test the abstract base class functionality"""

    def test_cannot_instantiate_base_class(self):
        """Test that BaseWSIInference cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            BaseWSIInference()

    def test_abstract_methods_required(self):
        """Test that all abstract methods must be implemented"""

        class IncompleteInference(BaseWSIInference):
            """Incomplete implementation missing abstract methods"""

            pass

        with self.assertRaises(TypeError):
            IncompleteInference()


class TestMIDOGInference(unittest.TestCase):
    """Test MIDOG-specific inference pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.inference = MIDOGInference()

    def test_initialization(self):
        """Test MIDOG inference initialization"""
        self.assertEqual(self.inference.patch_size, 512)
        self.assertEqual(self.inference.stride, 492)
        self.assertEqual(self.inference.resolution, 0.5)
        self.assertEqual(self.inference.units, "mpp")
        self.assertEqual(self.inference.post_proc_size, 21)
        self.assertEqual(self.inference.post_proc_threshold, 0.99)

    def test_model_config(self):
        """Test MIDOG model configuration"""
        config = self.inference.get_model_config()
        expected = {"num_heads": 1, "decoders_out_channels": [1]}
        self.assertEqual(config, expected)

    def test_target_channels(self):
        """Test MIDOG target channels"""
        channels = self.inference.get_target_channels()
        self.assertEqual(channels, [0])

    def test_cell_channel_map(self):
        """Test MIDOG cell channel mapping"""
        mapping = self.inference.get_cell_channel_map()
        expected = {"mitotic_figure": 0}
        self.assertEqual(mapping, expected)

    def test_output_suffix(self):
        """Test MIDOG output suffix"""
        suffix = self.inference.get_output_suffix()
        self.assertEqual(suffix, "_mitosis")

    def test_process_model_output(self):
        """Test MIDOG model output processing"""
        # Create mock probability tensors
        batch_size = 2
        probs = torch.randn(batch_size, 10, 256, 256)  # 10 channels
        prob_tensors = [torch.zeros(batch_size, 1, 256, 256)]

        # Process the output
        self.inference.process_model_output(probs, prob_tensors, batch_size)

        # Check that the tensor was modified
        expected_channel = self.inference.get_target_channels()[0]  # Channel 2
        expected_tensor = probs[:, expected_channel : expected_channel + 1, :, :]
        torch.testing.assert_close(prob_tensors[0], expected_tensor)


class TestPanNukeInference(unittest.TestCase):
    """Test PanNuke-specific inference pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.inference = PanNukeInference()

    def test_initialization(self):
        """Test PanNuke inference initialization"""
        self.assertEqual(self.inference.patch_size, 256)
        self.assertEqual(self.inference.stride, 240)
        self.assertEqual(self.inference.resolution, 0.25)
        self.assertEqual(self.inference.units, "mpp")
        self.assertEqual(self.inference.post_proc_size, 11)
        self.assertEqual(self.inference.post_proc_threshold, 0.5)

    def test_model_config(self):
        """Test PanNuke model configuration"""
        config = self.inference.get_model_config()
        expected = {"num_heads": 6, "decoders_out_channels": [3, 3, 3, 3, 3, 3]}
        self.assertEqual(config, expected)

    def test_target_channels(self):
        """Test PanNuke target channels"""
        channels = self.inference.get_target_channels()
        expected = [5, 8, 11, 14, 17]
        self.assertEqual(channels, expected)

    def test_cell_channel_map(self):
        """Test PanNuke cell channel mapping"""
        mapping = self.inference.get_cell_channel_map()
        expected = {
            "neoplastic": 0,
            "inflammatory": 1,
            "connective": 2,
            "dead": 3,
            "epithelial": 4,
        }
        self.assertEqual(mapping, expected)

    def test_output_suffix(self):
        """Test PanNuke output suffix"""
        suffix = self.inference.get_output_suffix()
        self.assertEqual(suffix, "_pannuke")

    def test_process_model_output(self):
        """Test PanNuke model output processing"""
        batch_size = 2
        probs = torch.randn(batch_size, 20, 256, 256)  # 20 channels
        prob_tensors = [torch.zeros(batch_size, 1, 256, 256) for _ in range(6)]

        # Process the output
        self.inference.process_model_output(probs, prob_tensors, batch_size)

        # Check that all tensors were modified correctly
        target_channels = self.inference.get_target_channels()
        for i, c in enumerate(target_channels):
            expected_tensor = probs[:, c : c + 1, :, :]
            torch.testing.assert_close(prob_tensors[i], expected_tensor)


class TestCoNICInference(unittest.TestCase):
    """Test CoNIC-specific inference pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.inference = CoNICInference()

    def test_initialization(self):
        """Test CoNIC inference initialization"""
        self.assertEqual(self.inference.patch_size, 256)
        self.assertEqual(self.inference.stride, 248)
        self.assertEqual(self.inference.resolution, 0.5)
        self.assertEqual(self.inference.units, "mpp")
        self.assertEqual(self.inference.post_proc_size, 5)
        self.assertEqual(self.inference.post_proc_threshold, 0.5)

    def test_model_config(self):
        """Test CoNIC model configuration"""
        config = self.inference.get_model_config()
        expected = {"num_heads": 6, "decoders_out_channels": [3, 3, 3, 3, 3, 3]}
        self.assertEqual(config, expected)

    def test_target_channels(self):
        """Test CoNIC target channels"""
        channels = self.inference.get_target_channels()
        expected = [2, 5, 8, 11, 14, 17]
        self.assertEqual(channels, expected)

    def test_cell_channel_map(self):
        """Test CoNIC cell channel mapping"""
        mapping = self.inference.get_cell_channel_map()
        expected = {
            "neutrophil": 0,
            "epithelial": 1,
            "lymphocyte": 2,
            "plasma": 3,
            "eosinophil": 4,
            "connective": 5,
        }
        self.assertEqual(mapping, expected)

    def test_output_suffix(self):
        """Test CoNIC output suffix"""
        suffix = self.inference.get_output_suffix()
        self.assertEqual(suffix, "_conic")


class TestMONKEYInference(unittest.TestCase):
    """Test MONKEY-specific inference pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.inference = MONKEYInference()

    def test_initialization(self):
        """Test MONKEY inference initialization"""
        self.assertEqual(self.inference.patch_size, 256)
        self.assertEqual(self.inference.stride, 224)
        self.assertEqual(self.inference.resolution, 0.5)
        self.assertEqual(self.inference.units, "mpp")
        self.assertEqual(self.inference.post_proc_size, 11)
        self.assertEqual(self.inference.post_proc_threshold, 0.5)

    def test_model_config(self):
        """Test MONKEY model configuration"""
        config = self.inference.get_model_config()
        expected = {
            "num_heads": 3,
            "decoders_out_channels": [3, 3, 3],
            "wide_decoder": True,
        }
        self.assertEqual(config, expected)

    def test_target_channels(self):
        """Test MONKEY target channels"""
        channels = self.inference.get_target_channels()
        expected = [2, 5, 8]
        self.assertEqual(channels, expected)

    def test_cell_channel_map(self):
        """Test MONKEY cell channel mapping"""
        mapping = self.inference.get_cell_channel_map()
        expected = {
            "overall_inflammatory": 0,
            "lymphocyte": 1,
            "monocyte": 2,
        }
        self.assertEqual(mapping, expected)

    def test_output_suffix(self):
        """Test MONKEY output suffix"""
        suffix = self.inference.get_output_suffix()
        self.assertEqual(suffix, "_monkey")

    def test_process_model_output_complex(self):
        """Test MONKEY's complex model output processing logic"""
        batch_size = 2
        probs = torch.randn(batch_size, 12, 256, 256)  # 12 channels
        prob_tensors = [torch.zeros(batch_size, 1, 256, 256) for _ in range(3)]

        # Process the output
        self.inference.process_model_output(probs, prob_tensors, batch_size)

        # Verify the complex processing logic was applied
        target_channels = self.inference.get_target_channels()
        for i, c in enumerate(target_channels):
            # MONKEY combines segmentation and centroid probabilities
            centroid_probs = probs[:, c : c + 1, :, :]
            seg_probs = probs[:, c - 2 : c - 1, :, :]

            # Check that processing occurred (tensor should not be zero)
            self.assertFalse(
                torch.allclose(prob_tensors[i], torch.zeros_like(prob_tensors[i]))
            )


class TestInferenceIntegration(unittest.TestCase):
    """Integration tests for inference pipeline components"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)

    def test_all_pipelines_have_consistent_interface(self):
        """Test that all inference pipelines implement the same interface"""
        pipelines = [MIDOGInference, PanNukeInference, CoNICInference, MONKEYInference]

        for pipeline_class in pipelines:
            pipeline = pipeline_class()

            # Test all abstract methods are implemented
            self.assertIsInstance(pipeline.get_model_config(), dict)
            self.assertIsInstance(pipeline.get_target_channels(), list)
            self.assertIsInstance(pipeline.get_cell_channel_map(), dict)
            self.assertIsInstance(pipeline.get_output_suffix(), str)

            # Test process_model_output method exists and is callable
            self.assertTrue(hasattr(pipeline, "process_model_output"))
            self.assertTrue(callable(getattr(pipeline, "process_model_output")))

    def test_model_config_consistency(self):
        """Test that model configurations have consistent structure"""
        pipelines = [
            MIDOGInference(),
            PanNukeInference(),
            CoNICInference(),
            MONKEYInference(),
        ]

        for pipeline in pipelines:
            config = pipeline.get_model_config()

            # Check required keys
            self.assertIn("num_heads", config)
            self.assertIn("decoders_out_channels", config)

            # Check data types
            self.assertIsInstance(config["num_heads"], int)
            self.assertIsInstance(config["decoders_out_channels"], list)

            # Check consistency between num_heads and decoders_out_channels length
            self.assertEqual(config["num_heads"], len(config["decoders_out_channels"]))

    def test_channel_mapping_consistency(self):
        """Test that channel mappings are consistent with target channels"""
        pipelines = [
            MIDOGInference(),
            PanNukeInference(),
            CoNICInference(),
            MONKEYInference(),
        ]

        for pipeline in pipelines:
            target_channels = pipeline.get_target_channels()
            cell_channel_map = pipeline.get_cell_channel_map()

            # Number of target channels should match number of cell types
            self.assertEqual(len(target_channels), len(cell_channel_map))

            # Channel indices should be sequential starting from 0
            expected_indices = list(range(len(cell_channel_map)))
            actual_indices = sorted(cell_channel_map.values())
            self.assertEqual(expected_indices, actual_indices)


if __name__ == "__main__":
    unittest.main()
