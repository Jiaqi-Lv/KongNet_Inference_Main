import logging

from inference.base_inference_interface import BaseInferenceInterface
from inference.wsi_inference_MIDOG import MIDOGInference

logger = logging.getLogger()
logger.disabled = True


def main():
    """Main entry point for MIDOG inference"""
    interface = BaseInferenceInterface(
        inference_class=MIDOGInference,
        pipeline_name="MIDOG",
        default_hf_repo="TIACentre/KongNet_pretrained_weights",
        default_checkpoint="KongNet_Det_MIDOG_1.pth",
    )
    interface.main()


if __name__ == "__main__":
    main()
