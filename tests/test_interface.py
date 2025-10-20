"""
Unit tests for the base inference interface and command line handling.

Tests the BaseInferenceInterface class that handles argument parsing,
model loading, and pipeline orchestration.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import os
import shutil
import argparse

from inference.base_inference_interface import BaseInferenceInterface
from inference.wsi_inference_CoNIC import CoNICInference


class TestBaseInferenceInterface(unittest.TestCase):
    """Test the BaseInferenceInterface class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.interface = BaseInferenceInterface(
            inference_class=CoNICInference,
            pipeline_name="TestPipeline", 
            default_hf_repo="test/repo",
            default_checkpoint="test_checkpoint.pth"
        )
        
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test BaseInferenceInterface initialization"""
        self.assertEqual(self.interface.pipeline_name, "TestPipeline")
        self.assertEqual(self.interface.default_hf_repo, "test/repo")
        self.assertEqual(self.interface.default_checkpoint, "test_checkpoint.pth")
        self.assertIsInstance(self.interface.inference_instance, CoNICInference)
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_defaults(self, mock_parse_args):
        """Test argument parsing with default values"""
        # Mock the argument parser return
        mock_args = argparse.Namespace()
        mock_args.input_dir = "./test_input"
        mock_args.output_dir = "./test_output"
        mock_args.cache_dir = "/tmp/cache"
        mock_args.weights_dir = "./model_weights"
        mock_args.hf_repo_id = "test/repo"
        mock_args.checkpoint_name = "test_checkpoint.pth"
        mock_args.additional_checkpoints = None
        mock_args.local_weights = None
        mock_args.no_tta = False
        mock_args.single_wsi = None
        mock_args.mask_dir = None
        mock_args.num_workers = 10
        mock_args.batch_size = 64
        
        mock_parse_args.return_value = mock_args
        
        args = self.interface.parse_args()
        
        self.assertEqual(args.hf_repo_id, "test/repo")
        self.assertEqual(args.checkpoint_name, "test_checkpoint.pth")
        self.assertEqual(args.num_workers, 10)
        self.assertEqual(args.batch_size, 64)
    
    def test_get_wsi_list_all_files(self):
        """Test getting WSI file list from directory"""
        # Create test WSI files
        test_files = ["sample1.svs", "sample2.tif", "sample3.ndpi", "sample4.mrxs", "readme.txt"]
        for filename in test_files:
            with open(os.path.join(self.temp_dir, filename), 'w') as f:
                f.write("test")
        
        # Mock args
        args = Mock()
        args.input_dir = self.temp_dir
        args.single_wsi = None
        
        wsi_list = self.interface.get_wsi_list(args)
        
        # Should only include WSI files, not txt files
        expected_files = ["sample1.svs", "sample2.tif", "sample3.ndpi", "sample4.mrxs"]
        self.assertEqual(sorted(wsi_list), sorted(expected_files))
    
    def test_get_wsi_list_single_wsi(self):
        """Test getting single WSI file"""
        # Create test file
        test_file = "sample.svs"
        with open(os.path.join(self.temp_dir, test_file), 'w') as f:
            f.write("test")
        
        # Mock args
        args = Mock()
        args.input_dir = self.temp_dir
        args.single_wsi = test_file
        
        wsi_list = self.interface.get_wsi_list(args)
        
        self.assertEqual(wsi_list, [test_file])
    
    def test_get_wsi_list_single_wsi_not_found(self):
        """Test handling of non-existent single WSI file"""
        # Mock args
        args = Mock()
        args.input_dir = self.temp_dir
        args.single_wsi = "nonexistent.svs"
        
        wsi_list = self.interface.get_wsi_list(args)
        
        self.assertEqual(wsi_list, [])


if __name__ == '__main__':
    unittest.main()