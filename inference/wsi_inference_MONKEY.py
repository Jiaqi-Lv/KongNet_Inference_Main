from inference.wsi_inference_base import BaseWSIInference
import torch

class MONKEYInference(BaseWSIInference):
    """MONKEY-specific inference pipeline"""

    def __init__(self):
        super().__init__()
        # MONKEY-specific configuration
        self.patch_size = 256
        self.stride = 224
        self.resolution = 0.5
        self.units = 'mpp'
        self.post_proc_size = 11
        self.post_proc_threshold = 0.5
    
    def get_model_config(self):
        """Return MONKEY model configuration"""
        return {
            "num_heads": 3,
            "decoders_out_channels": [3, 3, 3]
        }
    
    def get_target_channels(self):
        """Return MONKEY target channels"""
        return [2, 5, 8]
    
    def get_cell_channel_map(self):
        """Return MONKEY cell channel mapping"""
        return {
            "overall_inflammatory": 0,
            "lymphocyte": 1,
            "monocyte": 2,
        }
    
    def get_output_suffix(self):
        """Return MONKEY output suffix"""
        return "_monkey"
    
    def process_model_output(self, probs: torch.Tensor, prob_tensors: list, batch_size: int):
        """Process model output for MONKEY - combines segmentation and centroid probabilities"""
        selected_channels = self.get_target_channels()
        for i, c in enumerate(selected_channels):
            _centroid_probs = probs[:, c : c + 1, :, :]
            _seg_probs = probs[:, c - 2 : c - 1, :, :]
            _seg_probs *= (_centroid_probs >= 0.5).to(
                dtype=torch.float16
            )
            _centroid_probs = (
                _seg_probs * 0.4 + _centroid_probs * 0.6
            )
            prob_tensors[i] += _centroid_probs

# Create global instance and expose start function for backward compatibility
_monkey_inference = MONKEYInference()

def start(
    wsi_path: str,
    det_models: list,
    mask_path: str | None,
    save_dir: str | None,
    cache_dir: str = "./cache",
    num_workers: int = 10,
    batch_size: int = 64,
):
    """MONKEY inference entry point"""
    return _monkey_inference.start(
        wsi_path=wsi_path,
        det_models=det_models,
        mask_path=mask_path,
        save_dir=save_dir,
        cache_dir=cache_dir,
        num_workers=num_workers,
        batch_size=batch_size,
    )