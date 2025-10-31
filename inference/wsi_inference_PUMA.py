import torch

from inference.wsi_inference_base import BaseWSIInference


class PUMAInference(BaseWSIInference):
    """PUMA-specific inference pipeline"""

    def __init__(self):
        super().__init__()
        # PUMA-specific configuration
        self.patch_size = 256
        self.stride = 224
        self.resolution = 0.25
        self.units = "mpp"
        self.post_proc_size = 13
        self.nms_box_size = 15
        self.post_proc_threshold = 0.5
        self.track_id = 1  # Default to track 1; can be set to 2 for track 2

    def set_track(self, track_id: int):
        """Set the PUMA track ID (1 or 2)"""
        if track_id not in [1, 2]:
            raise ValueError("track_id must be either 1 or 2")
        self.track_id = track_id

    def get_model_config(self):
        """Return PUMA model configuration"""
        num_heads = 3 if self.track_id == 1 else 10
        out_channels = [3] * num_heads
        return {
            "num_heads": num_heads,
            "decoders_out_channels": out_channels,
            "wide_decoder": False,
        }

    def get_target_channels(self):
        """Return PUMA target channels"""
        if self.track_id == 1:
            return [2, 5, 8]
        else:
            return [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]

    def get_cell_channel_map(self):
        """Return PUMA cell channel mapping"""
        if self.track_id == 1:
            return {
                "tumour_cell": 0,
                "lymphocyte": 1,
                "other_cell": 2,
            }
        else:
            return {
                "tumour_cell": 0,
                "lymphocyte": 1,
                "plasma_cell": 2,
                "histiocyte": 3,
                "melanophage": 4,
                "neutrophil": 5,
                "stroma_cell": 6,
                "epithelial_cell": 7,
                "endothelial_cell": 8,
                "apoptotic_cell": 9,
            }

    def get_output_suffix(self):
        """Return PUMA output suffix"""
        return f"_puma_T{self.track_id}"

    def process_model_output(
        self, probs: torch.Tensor, prob_tensors: list, batch_size: int
    ):
        """Process model output for PUMA - combines segmentation and centroid probabilities"""
        selected_channels = self.get_target_channels()
        for i, c in enumerate(selected_channels):
            _centroid_probs = probs[:, c : c + 1, :, :]
            _seg_probs = probs[:, c - 2 : c - 1, :, :]
            _seg_probs *= (_centroid_probs >= 0.5).to(dtype=torch.float16)
            _centroid_probs = _seg_probs * 0.4 + _centroid_probs * 0.6
            prob_tensors[i] += _centroid_probs
