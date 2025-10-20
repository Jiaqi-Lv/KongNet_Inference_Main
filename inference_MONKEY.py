import logging

from inference.base_inference_interface import BaseInferenceInterface
from inference.wsi_inference_MONKEY import MONKEYInference

logger = logging.getLogger()
logger.disabled = True


def main():
    """Main entry point for MONKEY inference"""
    interface = BaseInferenceInterface(
        inference_class=MONKEYInference,
        pipeline_name="MONKEY",
        default_hf_repo="TIACentre/KongNet_pretrained_weights",
        default_checkpoint="KongNet_MONKEY_1.pth",
    )
    interface.main()


if __name__ == "__main__":
    main()
