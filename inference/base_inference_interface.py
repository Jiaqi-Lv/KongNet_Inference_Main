import argparse
import logging
import os
import shutil
import time

import torch
import ttach as tta
from tqdm.auto import tqdm

from inference.data_utils import download_weights_from_hf
from inference.wsi_inference_base import BaseWSIInference
from model.KongNet import get_KongNet

logger = logging.getLogger()
logger.disabled = True


class BaseInferenceInterface:
    """Base class for inference CLI interfaces"""

    def __init__(
        self,
        inference_class: BaseWSIInference,
        pipeline_name: str,
        default_hf_repo: str,
        default_checkpoint: str,
    ):
        self.inference_class = inference_class
        self.pipeline_name = pipeline_name
        self.default_hf_repo = default_hf_repo
        self.default_checkpoint = default_checkpoint
        self.inference_instance = inference_class()

    def parse_args(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description=f"KongNet {self.pipeline_name} Inference Pipeline",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Input/Output paths
        parser.add_argument(
            "--input_dir",
            type=str,
            default="/home/u1910100/Github/KongNet_Inference_Main/test_input",
            help="Directory containing input WSI files",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="/home/u1910100/Github/KongNet_Inference_Main/test_output",
            help="Directory to save inference results",
        )
        parser.add_argument(
            "--cache_dir",
            type=str,
            default="/home/u1910100/cloud_workspace/data/cache",
            help="Directory for caching intermediate results",
        )
        parser.add_argument(
            "--weights_dir",
            type=str,
            default="./model_weights",
            help="Directory to store model weights",
        )

        # Model configuration
        parser.add_argument(
            "--hf_repo_id",
            type=str,
            default=self.default_hf_repo,
            help="Hugging Face repository ID for model weights",
        )
        parser.add_argument(
            "--checkpoint_name",
            type=str,
            default=self.default_checkpoint,
            help="Name of the checkpoint file",
        )
        parser.add_argument(
            "--additional_checkpoints",
            nargs="+",
            default=None,
            help="Additional checkpoint files for model ensemble (space-separated)",
        )
        parser.add_argument(
            "--local_weights",
            nargs="+",
            default=None,
            help="Local paths to model weights (bypass HF download)",
        )

        # Processing options
        parser.add_argument(
            "--no_tta", action="store_true", help="Disable test time augmentation"
        )
        parser.add_argument(
            "--single_wsi",
            type=str,
            default=None,
            help="Process only a single WSI file (specify filename only)",
        )
        parser.add_argument(
            "--mask_dir",
            type=str,
            default=None,
            help="Directory containing mask files (optional)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=10,
            help="Number of worker processes for data loading and processing",
        )
        parser.add_argument(
            "--batch_size", type=int, default=64, help="Batch size for inference"
        )

        return parser.parse_args()

    def load_models(self, args):
        """Load and prepare models based on arguments"""
        model_weight_paths = []

        if args.local_weights:
            # Use local weight paths
            model_weight_paths = args.local_weights
            print(f"Using local model weights: {model_weight_paths}")
        else:
            # Download from Hugging Face
            print(f"Downloading weights from Hugging Face repo: {args.hf_repo_id}")
            os.makedirs(args.weights_dir, exist_ok=True)

            # Download main checkpoint
            checkpoint_path = download_weights_from_hf(
                checkpoint_name=args.checkpoint_name,
                repo_id=args.hf_repo_id,
                save_dir=args.weights_dir,
            )
            model_weight_paths.append(checkpoint_path)

            # Download additional checkpoints if specified
            if args.additional_checkpoints:
                for checkpoint_name in args.additional_checkpoints:
                    checkpoint_path = download_weights_from_hf(
                        checkpoint_name=checkpoint_name,
                        repo_id=args.hf_repo_id,
                        save_dir=args.weights_dir,
                    )
                    model_weight_paths.append(checkpoint_path)

        # Load models
        det_models = []
        model_config = self.inference_instance.get_model_config()

        for weight_path in model_weight_paths:
            if not args.no_tta:
                transforms = tta.Compose(
                    [
                        tta.Rotate90(angles=[0, 90, 180, 270]),
                    ]
                )

            model = get_KongNet(
                num_heads=model_config["num_heads"],
                decoders_out_channels=model_config["decoders_out_channels"],
                wide_decoder=model_config.get("wide_decoder", False),
            )

            checkpoint = torch.load(weight_path, map_location="cuda")
            print(
                f"Loading checkpoint from {weight_path}, epoch: {checkpoint.get('epoch', 'unknown')}"
            )
            model.load_state_dict(checkpoint["model"])
            model.eval()

            if not args.no_tta:
                model = tta.SegmentationTTAWrapper(model, transforms)

            model.to("cuda")
            det_models.append(model)

        print(f"Loaded {len(det_models)} model(s) on CUDA")
        return det_models

    def get_wsi_list(self, args):
        """Get list of WSI files to process"""
        if args.single_wsi:
            # Single WSI specified - should be just filename
            if os.path.exists(os.path.join(args.input_dir, args.single_wsi)):
                wsi_list = [args.single_wsi]
                print(f"Processing single WSI: {args.single_wsi}")
            else:
                print(
                    f"Error: WSI file '{args.single_wsi}' not found in {args.input_dir}"
                )
                return []
        else:
            wsi_list = [
                f
                for f in os.listdir(args.input_dir)
                if f.lower().endswith((".svs", ".tif", ".tiff", ".ndpi", ".mrxs"))
            ]
            print(f"Found {len(wsi_list)} WSI files in {args.input_dir}")

        return wsi_list

    def main(self):
        """Main inference pipeline"""
        args = self.parse_args()

        print(f"Configuration:")
        print(f"  Pipeline: {self.pipeline_name}")
        print(f"  Input directory: {args.input_dir}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Cache directory: {args.cache_dir}")
        print(f"  Number of workers: {args.num_workers}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  TTA enabled: {not args.no_tta}")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load models
        det_models = self.load_models(args)

        # Get WSI list
        wsi_list = self.get_wsi_list(args)

        if not wsi_list:
            print("No WSI files found to process.")
            return

        # Process WSIs
        total_time = 0
        successful_count = 0

        for wsi_name in tqdm(wsi_list, desc="Processing WSIs", total=len(wsi_list)):
            wsi_name_without_ext = os.path.splitext(wsi_name)[0]

            wsi_path = os.path.join(args.input_dir, wsi_name)

            # Check for corresponding mask file
            mask_path = None
            if args.mask_dir:
                mask_candidates = [
                    os.path.join(args.mask_dir, f"{wsi_name_without_ext}.png"),
                    os.path.join(args.mask_dir, f"{wsi_name_without_ext}.npy"),
                    os.path.join(args.mask_dir, f"{wsi_name_without_ext}.tif"),
                    os.path.join(args.mask_dir, f"{wsi_name_without_ext}.tiff"),
                ]
                for candidate in mask_candidates:
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break

            wsi_cache_dir = os.path.join(args.cache_dir, wsi_name_without_ext)

            print(f"\nProcessing: {wsi_name}")
            print(f"WSI path: {wsi_path}")
            print(f"Mask path: {mask_path}")
            print(f"Output dir: {args.output_dir}")
            print(f"Cache dir: {wsi_cache_dir}")

            try:
                start_time = time.time()
                self.inference_instance.start(
                    wsi_path=wsi_path,
                    det_models=det_models,
                    mask_path=mask_path,
                    save_dir=args.output_dir,
                    cache_dir=wsi_cache_dir,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    weights_dir=args.weights_dir,
                )
                end_time = time.time()

                processing_time = end_time - start_time
                total_time += processing_time
                successful_count += 1

                print(f"Detection completed in {processing_time:.2f} seconds")

                # Always clean up cache
                if os.path.exists(wsi_cache_dir):
                    shutil.rmtree(wsi_cache_dir)
                    print(f"Cache directory cleaned: {wsi_cache_dir}")

            except Exception as e:
                print(f"✗ Error processing {wsi_name}: {e}")
                # Clean up cache on error
                if os.path.exists(wsi_cache_dir):
                    shutil.rmtree(wsi_cache_dir)
                    print(f"Cache directory removed due to error: {wsi_cache_dir}")
                continue

        # Summary
        print(f"\n{'='*50}")
        print(f"Processing complete!")
        print(f"Successfully processed: {successful_count}/{len(wsi_list)} WSIs")
        print(f"Total processing time: {total_time:.2f} seconds")
        if successful_count > 0:
            print(f"Average time per WSI: {total_time/successful_count:.2f} seconds")
        print(f"Results saved to: {args.output_dir}")
