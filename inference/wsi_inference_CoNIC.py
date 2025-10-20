from inference.wsi_inference_base import BaseWSIInference


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


# Create global instance and expose start function for backward compatibility
_conic_inference = CoNICInference()

def start(
    wsi_path: str,
    det_models: list,
    mask_path: str | None,
    save_dir: str | None,
    cache_dir: str = "./cache",
    num_workers: int = 10,
    batch_size: int = 64,
):
    """CoNIC inference entry point"""
    return _conic_inference.start(
        wsi_path=wsi_path,
        det_models=det_models,
        mask_path=mask_path,
        save_dir=save_dir,
        cache_dir=cache_dir,
        num_workers=num_workers,
        batch_size=batch_size,
    )