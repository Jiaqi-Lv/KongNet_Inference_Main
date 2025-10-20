from inference.wsi_inference_base import BaseWSIInference
import torch


class PanNukeInference(BaseWSIInference):
    """PanNuke-specific inference pipeline"""
    
    def __init__(self):
        super().__init__()
        # PanNuke-specific configuration
        self.patch_size = 256
        self.stride = 240
        self.resolution = 0.25
        self.units = 'mpp'
        self.post_proc_size = 11 # or 9
        self.post_proc_threshold = 0.5
    
    def get_model_config(self):
        """Return PanNuke model configuration"""
        return {
            "num_heads": 6,
            "decoders_out_channels": [3, 3, 3, 3, 3, 3]
        }
    
    def get_target_channels(self):
        """Return PanNuke target channels"""
        return [2, 5, 8, 11, 14, 17]
    
    def get_cell_channel_map(self):
        """Return PanNuke cell channel mapping"""
        return {
            "overall": 0,
            "neoplastic": 1,
            "inflammatory": 2,
            "connective": 3,
            "dead": 4,
            "epithelial": 5,
        }
    
    def get_output_suffix(self):
        """Return PanNuke output suffix"""
        return "_pannuke"
    
    def process_model_output(self, probs: torch.Tensor, prob_tensors: list, batch_size: int):
        """Process model output for PanNuke - simple channel selection"""
        selected_channels = self.get_target_channels()
        for i, c in enumerate(selected_channels):
            prob_tensors[i] += probs[:, c:c+1, :, :]
