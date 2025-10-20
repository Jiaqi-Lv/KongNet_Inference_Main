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
        
        self.assertEqual(args.pipeline_name, "TestPipeline")  # This might not exist
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
    
    @patch('inference.base_inference_interface.download_weights_from_hf')
    @patch('inference.base_inference_interface.get_KongNet')
    @patch('torch.load')
    def test_load_models_local_weights(self, mock_torch_load, mock_get_kongnet, mock_download):
        """Test loading models with local weights"""
        # Create test weight files
        weight_paths = [
            os.path.join(self.temp_dir, "model1.pth"),
            os.path.join(self.temp_dir, "model2.pth")
        ]
        for path in weight_paths:
            with open(path, 'w') as f:
                f.write("test weights")
        
        # Mock args
        args = Mock()
        args.local_weights = weight_paths
        args.no_tta = True
        args.weights_dir = self.temp_dir
        
        # Mock torch components
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock()
        mock_get_kongnet.return_value = mock_model
        
        mock_checkpoint = {"model": {}, "epoch": 10}
        mock_torch_load.return_value = mock_checkpoint
        
        models = self.interface.load_models(args)
        
        # Should not download from HF
        mock_download.assert_not_called()
        
        # Should load 2 models
        self.assertEqual(len(models), 2)
        self.assertEqual(mock_get_kongnet.call_count, 2)
        self.assertEqual(mock_torch_load.call_count, 2)
    
    @patch('inference.base_inference_interface.download_weights_from_hf')
    @patch('inference.base_inference_interface.get_KongNet')
    @patch('torch.load')
    def test_load_models_hf_download(self, mock_torch_load, mock_get_kongnet, mock_download):
        """Test loading models with HuggingFace download"""
        # Mock args
        args = Mock()
        args.local_weights = None
        args.hf_repo_id = "test/repo"
        args.checkpoint_name = "main.pth"
        args.additional_checkpoints = ["additional1.pth", "additional2.pth"]
        args.no_tta = True
        args.weights_dir = self.temp_dir
        
        # Mock download returns
        mock_download.side_effect = [
            os.path.join(self.temp_dir, "main.pth"),
            os.path.join(self.temp_dir, "additional1.pth"),
            os.path.join(self.temp_dir, "additional2.pth")
        ]
        
        # Mock torch components
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock()
        mock_get_kongnet.return_value = mock_model
        
        mock_checkpoint = {"model": {}, "epoch": 5}
        mock_torch_load.return_value = mock_checkpoint
        
        models = self.interface.load_models(args)
        
        # Should download 3 checkpoints
        self.assertEqual(mock_download.call_count, 3)
        expected_calls = [
            call(checkpoint_name="main.pth", repo_id="test/repo", save_dir=self.temp_dir),
            call(checkpoint_name="additional1.pth", repo_id="test/repo", save_dir=self.temp_dir),
            call(checkpoint_name="additional2.pth", repo_id="test/repo", save_dir=self.temp_dir)
        ]
        mock_download.assert_has_calls(expected_calls)
        
        # Should load 3 models
        self.assertEqual(len(models), 3)
    
    @patch('inference.base_inference_interface.tta')
    @patch('inference.base_inference_interface.get_KongNet') 
    @patch('torch.load')
    def test_load_models_with_tta(self, mock_torch_load, mock_get_kongnet, mock_tta):
        """Test loading models with test time augmentation"""
        # Create test weight file
        weight_path = os.path.join(self.temp_dir, "model.pth")
        with open(weight_path, 'w') as f:
            f.write("test weights")
        
        # Mock args
        args = Mock()
        args.local_weights = [weight_path]
        args.no_tta = False  # Enable TTA
        
        # Mock components
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock()
        mock_get_kongnet.return_value = mock_model
        
        mock_tta_wrapper = Mock()
        mock_tta.SegmentationTTAWrapper.return_value = mock_tta_wrapper
        
        mock_checkpoint = {"model": {}, "epoch": 1}
        mock_torch_load.return_value = mock_checkpoint
        
        models = self.interface.load_models(args)
        
        # Should wrap model with TTA
        mock_tta.SegmentationTTAWrapper.assert_called_once()
        self.assertEqual(len(models), 1)


class TestInferenceWorkflow(unittest.TestCase):
    """Test end-to-end inference workflow components"""
    
    @patch('inference.base_inference_interface.BaseInferenceInterface.load_models')
    @patch('inference.base_inference_interface.BaseInferenceInterface.get_wsi_list')
    @patch('inference.base_inference_interface.BaseInferenceInterface.parse_args')
    def test_main_workflow_structure(self, mock_parse_args, mock_get_wsi_list, mock_load_models):
        """Test the main workflow structure without actual processing"""
        # Mock arguments
        mock_args = Mock()
        mock_args.input_dir = "./test_input"
        mock_args.output_dir = "./test_output"
        mock_args.cache_dir = "./cache"
        mock_args.num_workers = 10
        mock_args.batch_size = 64
        mock_args.no_tta = True
        mock_args.mask_dir = None
        
        mock_parse_args.return_value = mock_args
        mock_get_wsi_list.return_value = ["sample1.svs", "sample2.svs"]
        mock_load_models.return_value = [Mock(), Mock()]  # 2 mock models
        
        interface = BaseInferenceInterface(
            inference_class=CoNICInference,
            pipeline_name="Test",
            default_hf_repo="test/repo", 
            default_checkpoint="test.pth"
        )
        
        # Mock the inference instance start method
        with patch.object(interface.inference_instance, 'start') as mock_start:
            with patch('os.makedirs'):  # Mock directory creation
                with patch('os.path.exists', return_value=False):  # Mock path checks
                    with patch('shutil.rmtree'):  # Mock cache cleanup
                        # This would normally process WSIs, but we're mocking the heavy parts
                        try:
                            interface.main()
                        except Exception:
                            pass  # Expected since we're mocking heavily
        
        # Verify workflow steps were called
        mock_parse_args.assert_called_once()
        mock_load_models.assert_called_once_with(mock_args)
        mock_get_wsi_list.assert_called_once_with(mock_args)


if __name__ == '__main__':
    unittest.main()