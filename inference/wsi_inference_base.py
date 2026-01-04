import concurrent.futures
import os
import shutil
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import skimage
import torch
import zarr
from numcodecs import Blosc
from PIL import Image
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from inference.data_utils import (
    collate_fn,
    detection_to_annotation_store,
    download_weights_from_hf,
    imagenet_normalise_torch,
    slide_nms,
)
from inference.GrandQC_model import process_single_slide
from inference.prediction_utils import binary_det_post_process


class BaseWSIInference(ABC):
    """Base class for WSI inference pipelines"""

    def __init__(self):
        # Default configuration - subclasses should override as needed
        self.patch_size = 256
        self.stride = 240
        self.resolution = 0.25
        self.units = "mpp"
        self.post_proc_size = 11
        self.post_proc_threshold = 0.5
        self.nms_box_size = 11
        self.nms_threshold = 0.5
        self.target_channels = []
        self.cell_channel_map = {}
        self.output_suffix = "_detection"

    @abstractmethod
    def get_model_config(self):
        """Return model configuration (num_heads, decoders_out_channels)"""
        pass

    @abstractmethod
    def get_target_channels(self):
        """Return list of target channels to extract from model output"""
        pass

    @abstractmethod
    def get_cell_channel_map(self):
        """Return mapping of cell types to channel indices"""
        pass

    @abstractmethod
    def get_output_suffix(self):
        """Return suffix for output files"""
        pass

    @abstractmethod
    def process_model_output(
        self, probs: torch.Tensor, prob_tensors: list, batch_size: int
    ):
        """Process model output probabilities for target channels (in-place operation)

        This method modifies the prob_tensors list in-place by accumulating processed
        probabilities. It does not return anything as the results are stored directly
        in the provided prob_tensors.

        Args:
            probs: Model output probabilities [B, C, H, W]
            prob_tensors: List of probability tensors to accumulate results (modified in-place)
            batch_size: Current batch size
        """
        pass

    def detection_in_wsi(
        self,
        wsi_reader: WSIReader,
        mask_thumbnail: np.ndarray,
        models: list[torch.nn.Module],
        cache_dir: str = "./cache",
        num_workers: int = 10,
        batch_size: int = 64,
    ):
        """Run inference on WSI patches"""
        start_time = time.time()

        print(f"Using {num_workers} workers for data processing")
        print(f"Using batch size of {batch_size} for inference")

        patch_extractor = SlidingWindowPatchExtractor(
            input_img=wsi_reader,
            input_mask=mask_thumbnail,
            patch_size=(self.patch_size, self.patch_size),
            stride=(self.stride, self.stride),
            resolution=self.resolution,
            units=self.units,
            min_mask_ratio=0.3,
        )

        dataloader = DataLoader(
            patch_extractor,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

        n_tiles = len(dataloader.dataset)
        n_classes = len(self.get_target_channels())
        tile_size = self.patch_size

        os.makedirs(cache_dir, exist_ok=True)

        z_preds_path = os.path.join(cache_dir, "predictions.zarr")
        z_coords_path = os.path.join(cache_dir, "coords.zarr")

        if os.path.exists(z_preds_path):
            shutil.rmtree(z_preds_path)
        if os.path.exists(z_coords_path):
            shutil.rmtree(z_coords_path)

        # Create Zarr stores
        z_preds = zarr.open(
            z_preds_path,
            mode="w",
            shape=(n_tiles, n_classes, tile_size, tile_size),
            chunks=(batch_size, n_classes, tile_size, tile_size),
            dtype="f4",
            compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
        )
        z_coords = zarr.open(
            z_coords_path,
            mode="w",
            shape=(n_tiles, 4),
            chunks=(batch_size, 4),
            dtype="i4",
            compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
        )

        def dump_results(data, z_preds, z_coords):
            pred_np, coords_np, idx = data
            B = pred_np.shape[0]
            z_preds[idx : idx + B, :, :] = pred_np
            z_coords[idx : idx + B] = coords_np
            return

        futures = []
        start_idx = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for imgs in tqdm(
                dataloader, desc="Predicting patches", total=len(dataloader)
            ):
                batch_size_actual = imgs.shape[0]
                imgs = torch.permute(imgs, (0, 3, 1, 2)).to("cuda").float() / 255
                imgs = imagenet_normalise_torch(imgs)

                # Preallocate probability maps
                prob_tensors = [
                    torch.zeros(
                        (batch_size_actual, 1, self.patch_size, self.patch_size),
                        device="cuda",
                    )
                    for _ in range(n_classes)
                ]

                with torch.no_grad():
                    for model in models:
                        with autocast(device_type="cuda"):
                            logits = model(imgs)
                            probs = torch.sigmoid(logits)

                        # Process model output using subclass-specific logic
                        self.process_model_output(
                            probs, prob_tensors, batch_size_actual
                        )

                predictions = (
                    torch.cat([p / len(models) for p in prob_tensors], dim=1)
                    .cpu()
                    .numpy()
                )

                coords = np.array(
                    patch_extractor.coordinate_list[
                        start_idx : start_idx + batch_size_actual
                    ]
                )

                futures.append(
                    executor.submit(
                        dump_results,
                        (predictions, coords, start_idx),
                        z_preds,
                        z_coords,
                    )
                )
                start_idx += batch_size_actual

            # Wait for threads and raise errors if any
            for future in concurrent.futures.as_completed(futures):
                future.result()

        end_time = time.time()
        print("Predictions saved to Zarr format.")
        print(f"Inference time: {end_time - start_time:.2f} seconds")

    def process_one_chunk_all_channels(
        self, pred_chunk, coord_chunk, threshold, min_distance, cell_channel_map
    ):
        """Process one chunk of predictions for all channels"""
        result = {f"{k}_points": [] for k in cell_channel_map}

        for cell_type, ch_idx in cell_channel_map.items():
            for j in range(pred_chunk.shape[0]):
                probs_map_patch = pred_chunk[j, ch_idx, :, :]
                x_start, y_start, x_end, y_end = coord_chunk[j]

                processed_mask = binary_det_post_process(
                    probs_map_patch,
                    threshold=threshold,
                    min_distance=min_distance,
                )

                prob_map_labels = skimage.measure.label(processed_mask)
                prob_map_stats = skimage.measure.regionprops(
                    prob_map_labels, intensity_image=probs_map_patch
                )

                for region in prob_map_stats:
                    centroid = region["centroid"]

                    c, r, confidence = (
                        centroid[1],
                        centroid[0],
                        region["mean_intensity"],
                    )
                    c1 = c + x_start
                    r1 = r + y_start

                    prediction_record = {
                        "x": c1,
                        "y": r1,
                        "type": cell_type,
                        "prob": float(confidence),
                    }

                    result[f"{cell_type}_points"].append(prediction_record)
        return result

    def process_detection_masks_chunked_mp_streamed(
        self,
        predictions,
        coordinate_list,
        threshold=0.5,
        min_distance=9,
        batch_size=64,
        num_workers=10,
    ):
        """Process detection masks using multiprocessing"""
        cell_channel_map = self.get_cell_channel_map()

        N = predictions.shape[0]
        final_results = {f"{k}_points": [] for k in cell_channel_map}

        with Pool(processes=num_workers) as pool:
            futures = []
            for i in range(0, N, batch_size):
                pred_chunk = predictions[i : i + batch_size, :, :, :]
                coord_chunk = coordinate_list[i : i + batch_size]

                args = (
                    np.array(pred_chunk),
                    np.array(coord_chunk),
                    threshold,
                    min_distance,
                    cell_channel_map,
                )
                futures.append(
                    pool.apply_async(self.process_one_chunk_all_channels, args)
                )

            for f in tqdm(futures, desc="Processing chunks"):
                result = f.get()
                for key, val in result.items():
                    final_results[key].extend(val)

        return final_results

    def post_process(
        self,
        wsi_reader: WSIReader,
        mask_thumbnail: np.ndarray,
        cache_dir: str = "./cache",
        batch_size: int = 64,
        num_workers: int = 10,
    ):
        """Post-process detection results"""
        z_preds_path = os.path.join(cache_dir, "predictions.zarr")
        z_coords_path = os.path.join(cache_dir, "coords.zarr")

        z_preds = zarr.open(z_preds_path, mode="r")
        z_coords = zarr.open(z_coords_path, mode="r")

        start_time = time.time()
        cell_type_points = self.process_detection_masks_chunked_mp_streamed(
            predictions=z_preds,
            coordinate_list=z_coords,
            threshold=self.post_proc_threshold,
            min_distance=self.post_proc_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        end_time = time.time()
        print(f"Detection processing time: {end_time - start_time:.2f} seconds")

        # Collect all records
        all_records = []
        for cell_type in self.get_cell_channel_map():
            all_records.extend(cell_type_points[f"{cell_type}_points"])

        print(f"Total {len(all_records)} cell records detected")

        start_time = time.time()
        final_records = slide_nms(
            wsi_reader=wsi_reader,
            binary_mask=mask_thumbnail,
            detection_record=all_records,
            detection_mpp=self.resolution,
            tile_size=4096,
            box_size=self.nms_box_size,
            overlap_thresh=self.nms_threshold,
            cache_dir=cache_dir,
            num_workers=num_workers,
        )
        end_time = time.time()
        print(f"After NMS, {len(final_records)} cell records remain")
        print(f"NMS time: {end_time - start_time:.2f} seconds")

        return {
            "cell_records": final_records,
        }

    def start(
        self,
        wsi_path: str,
        det_models: list[torch.nn.Module],
        mask_path: str | None,
        save_dir: str | None,
        cache_dir: str = "./cache",
        num_workers: int = 10,
        batch_size: int = 64,
        weights_dir: str = "./model_weights",
    ):
        """Main inference pipeline"""
        if save_dir is None:
            save_dir = os.path.dirname(wsi_path)
            print(f"save_dir is None, set to {save_dir}")
        else:
            os.makedirs(save_dir, exist_ok=True)

        wsi_name_without_ext = os.path.splitext(os.path.basename(wsi_path))[0]

        store_save_path = os.path.join(
            save_dir, f"{wsi_name_without_ext}{self.get_output_suffix()}.db"
        )
        if os.path.exists(store_save_path):
            print(
                f"Annotation store {store_save_path} already exists, skipping detection"
            )
            return

        cache_dir = os.path.join(cache_dir, wsi_name_without_ext)
        os.makedirs(cache_dir, exist_ok=True)

        wsi_reader = WSIReader.open(wsi_path)

        # Handle tissue mask
        if mask_path is None:
            hf_repo_id = "TIACentre/GrandQC_Tissue_Detection"
            checkpoint_name = "grandqc_tissue_detection.pth"
            print(
                f"Downloading GrandQC weights from Hugging Face repo: {hf_repo_id}, checkpoint: {checkpoint_name}"
            )
            os.makedirs(weights_dir, exist_ok=True)
            grandQC_weights_path = download_weights_from_hf(
                checkpoint_name=checkpoint_name,
                repo_id=hf_repo_id,
                save_dir=weights_dir,
            )
            tissue_mask = process_single_slide(wsi_path, grandQC_weights_path)
            mask_filename = f"{wsi_name_without_ext}_MASK.png"
            mask_path = os.path.join(save_dir, mask_filename)
            Image.fromarray(tissue_mask).save(mask_path)
        else:
            tissue_mask = VirtualWSIReader.open(mask_path)
            tissue_mask = tissue_mask.slide_thumbnail(resolution=0, units="level")

        tissue_mask = np.where(tissue_mask >= 1, 1, 0).astype(np.uint8)
        if tissue_mask.sum() == 0:
            print("Tissue mask is empty, skipping this WSI")
            return

        # Run inference
        self.detection_in_wsi(
            wsi_reader=wsi_reader,
            mask_thumbnail=tissue_mask,
            models=det_models,
            cache_dir=cache_dir,
            num_workers=num_workers,
            batch_size=batch_size,
        )

        print("Starting post processing...")
        start_time = time.time()
        detection_records = self.post_process(
            wsi_reader=wsi_reader,
            mask_thumbnail=tissue_mask,
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        end_time = time.time()
        print(f"Post processing completed in {end_time - start_time:.2f} seconds")

        if len(detection_records["cell_records"]) == 0:
            print("No cells detected.")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"Cache directory {cache_dir} removed")
            return

        print("Saving results...")
        start_time = time.time()
        print(wsi_reader.info.as_dict())
        base_mpp = wsi_reader.convert_resolution_units(
            input_res=0, input_unit="level", output_unit="mpp"
        )[0]
        scale_factor = self.resolution / base_mpp
        annotation_store = detection_to_annotation_store(
            detection_records["cell_records"],
            scale_factor=scale_factor,
            shape_type="point",
        )
        annotation_store.dump(store_save_path)

        end_time = time.time()
        print(f"Detection results saved to {save_dir}")
        print(f"Saving time: {end_time - start_time:.2f} seconds")

        # Clean up cache directory
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cache directory {cache_dir} removed")

        print(f"{len(detection_records['cell_records'])} cells detected")
        return
