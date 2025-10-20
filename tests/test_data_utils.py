"""
Unit tests for data utilities and helper functions.

Tests the various utility functions used throughout the inference pipeline
including data loading, image processing, and annotation handling.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import os
import shutil

# Note: These imports may fail if the dependencies aren't installed in the test environment
# In a real testing setup, you'd want to handle these gracefully or use test doubles
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from inference.data_utils import (
        collate_fn,
        imagenet_normalise_torch, 
        slide_nms,
        detection_to_annotation_store,
    )
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False

try:
    from inference.prediction_utils import binary_det_post_process
    PRED_UTILS_AVAILABLE = True
except ImportError:
    PRED_UTILS_AVAILABLE = False


class TestDataUtils(unittest.TestCase):
    """Test data utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    @unittest.skipIf(not DATA_UTILS_AVAILABLE, "Data utils not available")
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_collate_fn(self):
        """Test the custom collate function for batch processing"""
        # Create mock batch data
        batch_data = [
            np.random.rand(256, 256, 3),  # Sample 1
            np.random.rand(256, 256, 3),  # Sample 2
            np.random.rand(256, 256, 3),  # Sample 3
        ]
        
        result = collate_fn(batch_data)
        
        # Should return a tensor with batch dimension
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 256, 256, 3))
    
    @unittest.skipIf(not DATA_UTILS_AVAILABLE, "Data utils not available") 
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_imagenet_normalise_torch(self):
        """Test ImageNet normalization function"""
        # Create test tensor [B, C, H, W]
        test_tensor = torch.rand(2, 3, 224, 224)
        
        normalized = imagenet_normalise_torch(test_tensor)
        
        # Should maintain same shape
        self.assertEqual(normalized.shape, test_tensor.shape)
        
        # Check that normalization was applied (values should be different)
        self.assertFalse(torch.allclose(normalized, test_tensor))
        
        # Check that normalized values are in expected range for ImageNet
        # ImageNet normalization typically results in values roughly in [-2, 2] range
        self.assertTrue(normalized.min() >= -3.0)
        self.assertTrue(normalized.max() <= 3.0)
    
    @unittest.skipIf(not DATA_UTILS_AVAILABLE, "Data utils not available")
    def test_detection_to_annotation_store(self):
        """Test conversion of detections to annotation store format"""
        # Create mock detection records
        detection_records = [
            {"x": 100, "y": 200, "type": "mitotic_figure", "prob": 0.95},
            {"x": 300, "y": 400, "type": "mitotic_figure", "prob": 0.87},
            {"x": 500, "y": 600, "type": "lymphocyte", "prob": 0.92},
        ]
        
        # Mock the annotation store creation
        with patch('inference.data_utils.SQLiteStore') as mock_store_class:
            mock_store = Mock()
            mock_store_class.return_value = mock_store
            
            result_store = detection_to_annotation_store(
                detection_records, 
                scale_factor=1.0, 
                shape_type="point"
            )
            
            # Should create and return a store
            mock_store_class.assert_called_once()
            self.assertEqual(result_store, mock_store)


class TestPredictionUtils(unittest.TestCase):
    """Test prediction utility functions"""
    
    @unittest.skipIf(not PRED_UTILS_AVAILABLE, "Prediction utils not available")
    def test_binary_det_post_process(self):
        """Test binary detection post-processing"""
        # Create mock probability map
        prob_map = np.random.rand(256, 256)
        
        # Apply some structure to make it more realistic
        # Add some high-probability regions
        prob_map[100:110, 100:110] = 0.9
        prob_map[200:210, 200:210] = 0.8
        
        processed_mask = binary_det_post_process(
            prob_map,
            threshold=0.5,
            min_distance=5
        )
        
        # Should return a binary mask
        self.assertIsInstance(processed_mask, np.ndarray)
        self.assertEqual(processed_mask.dtype, bool)
        self.assertEqual(processed_mask.shape, prob_map.shape)
        
        # Should have some detections where probability was high
        self.assertTrue(np.any(processed_mask))


class TestImageProcessing(unittest.TestCase):
    """Test image processing components"""
    
    def test_patch_coordinates_generation(self):
        """Test patch coordinate generation logic"""
        # This tests the logic that would be in SlidingWindowPatchExtractor
        # We'll test our own implementation of the coordinate logic
        
        image_width, image_height = 1000, 1000
        patch_size = 256
        stride = 224
        
        # Generate coordinates manually (simulating the extractor)
        coordinates = []
        for y in range(0, image_height - patch_size + 1, stride):
            for x in range(0, image_width - patch_size + 1, stride):
                coordinates.append([x, y, x + patch_size, y + patch_size])
        
        # Should generate expected number of patches
        expected_x_patches = len(range(0, image_width - patch_size + 1, stride))
        expected_y_patches = len(range(0, image_height - patch_size + 1, stride))
        expected_total = expected_x_patches * expected_y_patches
        
        self.assertEqual(len(coordinates), expected_total)
        
        # First patch should start at (0, 0)
        self.assertEqual(coordinates[0], [0, 0, patch_size, patch_size])
        
        # Check patch dimensions
        for coord in coordinates:
            x1, y1, x2, y2 = coord
            self.assertEqual(x2 - x1, patch_size)
            self.assertEqual(y2 - y1, patch_size)


class TestNMSLogic(unittest.TestCase):
    """Test Non-Maximum Suppression logic"""
    
    def test_nms_removes_close_detections(self):
        """Test that NMS removes detections that are too close together"""
        # Create test detections with some that are very close
        detections = [
            {"x": 100, "y": 100, "type": "cell", "prob": 0.9},
            {"x": 102, "y": 101, "type": "cell", "prob": 0.8},  # Very close to first
            {"x": 200, "y": 200, "type": "cell", "prob": 0.85}, # Far from others
            {"x": 201, "y": 199, "type": "cell", "prob": 0.7},  # Close to third
        ]
        
        # Simple NMS implementation for testing
        def simple_nms(detections, threshold=5.0):
            """Simple NMS implementation for testing"""
            if not detections:
                return []
            
            # Sort by probability (highest first)
            sorted_dets = sorted(detections, key=lambda x: x["prob"], reverse=True)
            kept = []
            
            for det in sorted_dets:
                # Check if too close to any kept detection
                too_close = False
                for kept_det in kept:
                    dist = np.sqrt((det["x"] - kept_det["x"])**2 + (det["y"] - kept_det["y"])**2)
                    if dist < threshold:
                        too_close = True
                        break
                
                if not too_close:
                    kept.append(det)
            
            return kept
        
        result = simple_nms(detections, threshold=5.0)
        
        # Should keep the highest probability detection from each cluster
        self.assertEqual(len(result), 2)  # Should remove 2 close duplicates
        
        # Kept detections should be the higher probability ones
        probs = [det["prob"] for det in result]
        self.assertIn(0.9, probs)  # Highest prob from first cluster
        self.assertIn(0.85, probs)  # Highest prob from second cluster


class TestCacheManagement(unittest.TestCase):
    """Test caching and temporary file management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_cache_directory_creation(self):
        """Test cache directory creation and cleanup"""
        wsi_name = "test_sample"
        wsi_cache_dir = os.path.join(self.cache_dir, wsi_name)
        
        # Create cache directory
        os.makedirs(wsi_cache_dir, exist_ok=True)
        self.assertTrue(os.path.exists(wsi_cache_dir))
        
        # Create some cache files
        pred_file = os.path.join(wsi_cache_dir, "predictions.zarr")
        coord_file = os.path.join(wsi_cache_dir, "coords.zarr")
        
        os.makedirs(pred_file, exist_ok=True)
        os.makedirs(coord_file, exist_ok=True)
        
        self.assertTrue(os.path.exists(pred_file))
        self.assertTrue(os.path.exists(coord_file))
        
        # Cleanup
        shutil.rmtree(wsi_cache_dir)
        self.assertFalse(os.path.exists(wsi_cache_dir))
    
    def test_zarr_path_generation(self):
        """Test Zarr file path generation logic"""
        cache_dir = "/tmp/cache/sample1"
        
        z_preds_path = os.path.join(cache_dir, "predictions.zarr")
        z_coords_path = os.path.join(cache_dir, "coords.zarr")
        
        self.assertEqual(z_preds_path, "/tmp/cache/sample1/predictions.zarr")
        self.assertEqual(z_coords_path, "/tmp/cache/sample1/coords.zarr")


if __name__ == '__main__':
    unittest.main()