from inference.wsi_inference_base import BaseWSIInference
import torch


class CoNICInference(BaseWSIInference):
    """CoNIC-specific inference pipeline"""

    def __init__(self):
        super().__init__()
        # CoNIC-specific configuration
        self.patch_size = 256
        self.stride = 248
        self.resolution = 0.5
        self.units = 'mpp'
        self.post_proc_size = 5 # or 3
        self.post_proc_threshold = 0.5
    
    def get_model_config(self):
        """Return CoNIC model configuration"""
        return {
            "num_heads": 6,
            "decoders_out_channels": [3, 3, 3, 3, 3, 3]
        }
    
    def get_target_channels(self):
        """Return CoNIC target channels"""
        return [2, 5, 8, 11, 14, 17]
    
    def get_cell_channel_map(self):
        """Return CoNIC cell channel mapping"""
        return {
            "neutrophil": 0,
            "epithelial": 1,
            "lymphocyte": 2,
            "plasma": 3,
            "eosinophil": 4,
            "connective": 5,
        }
    
    def get_output_suffix(self):
        """Return CoNIC output suffix"""
        return "_conic"
    
    def process_model_output(self, probs: torch.Tensor, prob_tensors: list, batch_size: int):
        """Process model output for CoNIC - simple channel selection"""
        selected_channels = self.get_target_channels()
        for i, c in enumerate(selected_channels):
            prob_tensors[i] += probs[:, c:c+1, :, :]


