from inference.wsi_inference_MIDOG import start
from model.KongNet import get_KongNet
from tqdm.auto import tqdm
import os
import torch
import ttach as tta
import time
import shutil
from inference.data_utils import download_weights_from_hf

import logging
logger = logging.getLogger()
logger.disabled = True


if __name__ == "__main__":

    hf_repo_id = "TIACentre/KongNet_MIDOG"
    checkpoint_name = "KongNet_Det_MIDOG_1.pth"
    weights_dir = "./model_weights"
    print(f"Downloading weights from Hugging Face repo: {hf_repo_id}, checkpoint: {checkpoint_name}")
    os.makedirs(weights_dir, exist_ok=True)
    checkpoint_1_path = download_weights_from_hf(
        checkpoint_name=checkpoint_name,
        repo_id=hf_repo_id,
        save_dir=weights_dir
    )

    # If using ensemble, add more paths to the list
    model_weight_paths = [
        checkpoint_1_path,
    ]
    det_models = []
    for weight_path in model_weight_paths:
        transforms = tta.Compose(
            [
                tta.Rotate90(angles=[0, 180, 90, 270]),
            ]
        )
        model = get_KongNet(
            num_heads=1,
            decoders_out_channels=[1],
        )
        checkpoint = torch.load(weight_path)
        print(f"epoch: {checkpoint['epoch']}")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model = tta.SegmentationTTAWrapper(model, transforms)
        model.to("cuda")
        det_models.append(model)

    print("Models loaded")

    wsi_dir = "/home/u1910100/Github/KongNet_Inference_Main/test_input"
    save_dir = "/home/u1910100/Github/KongNet_Inference_Main/test_output"
    cache_dir = "/home/u1910100/cloud_workspace/data/cache"
    wsi_list = os.listdir(wsi_dir)


    for wsi_name in tqdm(wsi_list, desc="Processing WSIs", total=len(wsi_list)):
        wsi_name_without_ext = os.path.splitext(wsi_name)[0]

        wsi_path = os.path.join(wsi_dir, wsi_name)
        mask_path = None
        wsi_cache_dir = os.path.join(cache_dir, wsi_name_without_ext)

        print("Starting detection...")
        print(f"wsi_path: {wsi_path}")
        print(f"mask_path: {mask_path}")
        print(f"save_dir: {save_dir}")
        try:
            start_time = time.time()
            start(wsi_path, det_models, mask_path, save_dir, wsi_cache_dir)
            end_time = time.time()
            print(f"Detection completed in {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error processing {wsi_name}: {e}")
            if os.path.exists(wsi_cache_dir):
                shutil.rmtree(wsi_cache_dir)
                print(f"Cache directory {wsi_cache_dir} removed")
            continue
        
    print("All WSIs processed.")

