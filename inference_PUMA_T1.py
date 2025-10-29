import logging

from inference.base_inference_interface import BaseInferenceInterface
from inference.wsi_inference_PUMA import PUMAInference

logger = logging.getLogger()
logger.disabled = True


def main():
    """Main entry point for PUMA track 1 inference"""
    interface = BaseInferenceInterface(
        inference_class=PUMAInference(track_id=1),
        pipeline_name="PUMA",
        default_hf_repo="TIACentre/KongNet_pretrained_weights",
        default_checkpoint="KongNet_PUMA_T1_3.pth",
    )
    interface.main()


if __name__ == "__main__":
    main()
