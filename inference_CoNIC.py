from inference.wsi_inference_CoNIC import CoNICInference
from inference.base_inference_interface import BaseInferenceInterface

import logging
logger = logging.getLogger()
logger.disabled = True


def main():
    """Main entry point for CoNIC inference"""
    interface = BaseInferenceInterface(
        inference_class=CoNICInference,
        pipeline_name="CoNIC",
        default_hf_repo="TIACentre/KongNet_pretrained_weights",
        default_checkpoint="KongNet_CoNIC_1.pth"
    )
    interface.main()


if __name__ == "__main__":
    main()