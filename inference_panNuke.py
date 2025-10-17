from inference.wsi_inference_PanNuke import PanNukeInference
from inference.base_inference_interface import BaseInferenceInterface

import logging
logger = logging.getLogger()
logger.disabled = True


def main():
    """Main entry point for PanNuke inference"""
    interface = BaseInferenceInterface(
        inference_class=PanNukeInference,
        pipeline_name="PanNuke",
        default_hf_repo="TIACentre/KongNet_pretrained_weights",
        default_checkpoint="KongNet_PanNuke_1.pth"
    )
    interface.main()


if __name__ == "__main__":
    main()