from inference.wsi_inference_base import BaseWSIInference


class PanNukeInference(BaseWSIInference):
    """PanNuke-specific inference pipeline"""
    
    def __init__(self):
        super().__init__()
        # PanNuke-specific configuration
        self.patch_size = 256
        self.stride = 240
        self.resolution = 0.25
        self.units = 'mpp'
        self.post_proc_size = 11
        self.post_proc_threshold = 0.5
    
    def get_model_config(self):
        """Return PanNuke model configuration"""
        return {
            "num_heads": 6,
            "decoders_out_channels": [3, 3, 3, 3, 3, 3]
        }
    
    def get_target_channels(self):
        """Return PanNuke target channels"""
        return [5, 8, 11, 14, 17]
    
    def get_cell_channel_map(self):
        """Return PanNuke cell channel mapping"""
        return {
            "neoplastic": 0,
            "inflammatory": 1,
            "connective": 2,
            "dead": 3,
            "epithelial": 4,
        }
    
    def get_output_suffix(self):
        """Return PanNuke output suffix"""
        return "_pannuke"


# Create global instance and expose start function for backward compatibility
_pannuke_inference = PanNukeInference()

def start(
    wsi_path: str,
    det_models: list,
    mask_path: str | None,
    save_dir: str | None,
    cache_dir: str = "./cache",
    num_workers: int = 10,
    batch_size: int = 64,
):
    """PanNuke inference entry point"""
    return _pannuke_inference.start(
        wsi_path=wsi_path,
        det_models=det_models,
        mask_path=mask_path,
        save_dir=save_dir,
        cache_dir=cache_dir,
        num_workers=num_workers,
        batch_size=batch_size,
    )