from inference.wsi_inference_base import BaseWSIInference
import torch


class MIDOGInference(BaseWSIInference):
    """MIDOG-specific inference pipeline"""
    
    def __init__(self):
        super().__init__()
        # MIDOG-specific configuration
        self.patch_size = 512
        self.stride = 492
        self.resolution = 0.5
        self.units = 'mpp'
        self.post_proc_size = 21
        self.post_proc_threshold = 0.99
    
    def get_model_config(self):
        """Return MIDOG model configuration"""
        return {
            "num_heads": 1,
            "decoders_out_channels": [1]
        }
    
    def get_target_channels(self):
        """Return MIDOG target channels"""
        return [0]
    
    def get_cell_channel_map(self):
        """Return MIDOG cell channel mapping"""
        return {
            "mitotic_figure": 0,
        }
    
    def get_output_suffix(self):
        """Return MIDOG output suffix"""
        return "_mitosis"
    
    def process_model_output(self, probs: torch.Tensor, prob_tensors: list, batch_size: int):
        """Process model output for MIDOG - simple channel selection"""
        selected_channels = self.get_target_channels()
        for i, c in enumerate(selected_channels):
            prob_tensors[i] += probs[:, c:c+1, :, :]


# Create global instance and expose start function for backward compatibility
_midog_inference = MIDOGInference()

def start(
    wsi_path: str,
    det_models: list,
    mask_path: str | None,
    save_dir: str | None,
    cache_dir: str = "./cache",
    num_workers: int = 10,
    batch_size: int = 32,
):
    """MIDOG inference entry point"""
    return _midog_inference.start(
        wsi_path=wsi_path,
        det_models=det_models,
        mask_path=mask_path,
        save_dir=save_dir,
        cache_dir=cache_dir,
        num_workers=num_workers,
        batch_size=batch_size,
    )


