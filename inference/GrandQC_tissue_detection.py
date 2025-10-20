"""
Grand Challenge QC Tissue Detection Script

This script performs automated tissue detection on whole slide images (WSI)
using a pre-trained UNet++ model. It processes slides at 10 MPP resolution,
applies patch-based inference, and generates tissue masks.

Features:
- Handles variable slide sizes through patch-based processing
- Applies JPEG compression to match training conditions
- Generates color-coded tissue masks
- Supports CUDA acceleration

Dependencies:
- openslide-python
- opencv-python
- numpy
- torch
- PIL
- segmentation-models-pytorch
"""

import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from openslide import OpenSlide
from PIL import Image
from tiatoolbox.wsicore.wsireader import WSIReader


def to_tensor_x(x, **kwargs):
    """Convert image to tensor format (C, H, W)."""
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(image, preprocessing_fn):
    """Apply preprocessing to image and convert to tensor."""
    image = np.array(image)
    x = preprocessing_fn(image)
    return to_tensor_x(x)


def make_class_map(mask, class_colors):
    """Create RGB color map from class mask."""
    height, width = mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in enumerate(class_colors):
        mask_indices = mask == class_id
        rgb[mask_indices] = color

    return rgb


# Configuration constants
DEVICE = "cuda"
DETECTION_MPP = 10
PATCH_SIZE = 512
ENCODER = "timm-efficientnet-b0"
JPEG_QUALITY = 80

# Overlay transparency parameters
OVERLAY_IMAGE_WEIGHT = 0.7
OVERLAY_MASK_WEIGHT = 0.3

# Color mapping for tissue classes
CLASS_COLORS = [[50, 50, 250], [128, 128, 128]]  # BLUE: TISSUE  # GRAY: BACKGROUND

# Directory paths
SLIDE_DIR = "/media/u1910100/data/slides"
OUTPUT_DIR = "/media/u1910100/data/overlays"
MODEL_WEIGHT_PATH = "/media/u1910100/data/GrandQC_Tissue_Detection_MPP10.pth"


def setup_directories_and_model():
    """Initialize output directories and load the tissue detection model."""
    # Create output directory
    tis_det_dir_mask = os.path.join(OUTPUT_DIR, "tis_det_mask/")
    os.makedirs(tis_det_dir_mask, exist_ok=True)

    # Load preprocessing function and model
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, "imagenet")

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        classes=2,
        activation=None,
    )

    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    return tis_det_dir_mask, preprocessing_fn, model


def get_slide_names():
    """Get sorted list of slide files from the slide directory."""
    return sorted(
        [f for f in os.listdir(SLIDE_DIR) if os.path.isfile(os.path.join(SLIDE_DIR, f))]
    )


def apply_jpeg_compression(image):
    """Apply JPEG compression to reproduce training conditions."""
    image_array = np.array(image)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    _, compressed = cv2.imencode(".jpg", image_array, encode_param)
    return cv2.imdecode(compressed, 1)


def get_patch_crop_coordinates(w, h, width, height, wi_n, he_n):
    """Calculate crop coordinates for a patch based on position."""
    if w != wi_n and h != he_n:
        # Regular patch
        return (
            w * PATCH_SIZE,
            h * PATCH_SIZE,
            (w + 1) * PATCH_SIZE,
            (h + 1) * PATCH_SIZE,
        )
    elif w == wi_n and h != he_n:
        # Right edge patch
        return (width - PATCH_SIZE, h * PATCH_SIZE, width, (h + 1) * PATCH_SIZE)
    elif w != wi_n and h == he_n:
        # Bottom edge patch
        return (w * PATCH_SIZE, height - PATCH_SIZE, (w + 1) * PATCH_SIZE, height)
    else:
        # Bottom-right corner patch
        return (width - PATCH_SIZE, height - PATCH_SIZE, width, height)


def process_single_patch(patch_image, model, preprocessing_fn):
    """Process a single image patch through the model."""
    image_pre = get_preprocessing(patch_image, preprocessing_fn)
    x_tensor = torch.from_numpy(image_pre).to(DEVICE).unsqueeze(0)
    predictions = model.predict(x_tensor)
    predictions = predictions.squeeze().cpu().numpy()

    mask = np.argmax(predictions, axis=0).astype("int8")
    class_mask = make_class_map(mask, CLASS_COLORS)

    return mask, class_mask


def concatenate_patches_horizontally(
    w, wi_n, overhang_wi, mask, class_mask, temp_image, temp_image_class_map
):
    """Concatenate patches horizontally, handling edge cases."""
    if w == 0:
        return mask, class_mask
    elif w == wi_n:
        # Handle right edge overhang
        crop_start = PATCH_SIZE - overhang_wi
        mask_cropped = mask[:, crop_start:PATCH_SIZE]
        class_mask_cropped = class_mask[:, crop_start:PATCH_SIZE, :]

        return (
            np.concatenate((temp_image, mask_cropped), axis=1),
            np.concatenate((temp_image_class_map, class_mask_cropped), axis=1),
        )
    else:
        return (
            np.concatenate((temp_image, mask), axis=1),
            np.concatenate((temp_image_class_map, class_mask), axis=1),
        )


def concatenate_rows_vertically(
    h,
    he_n,
    overhang_he,
    temp_image,
    temp_image_class_map,
    end_image,
    end_image_class_map,
):
    """Concatenate row results vertically, handling edge cases."""
    if h == 0:
        return temp_image, temp_image_class_map
    elif h == he_n:
        # Handle bottom edge overhang
        crop_start = PATCH_SIZE - overhang_he
        temp_image_cropped = temp_image[crop_start:PATCH_SIZE, :]
        temp_class_cropped = temp_image_class_map[crop_start:PATCH_SIZE, :, :]

        return (
            np.concatenate((end_image, temp_image_cropped), axis=0),
            np.concatenate((end_image_class_map, temp_class_cropped), axis=0),
        )
    else:
        return (
            np.concatenate((end_image, temp_image), axis=0),
            np.concatenate((end_image_class_map, temp_image_class_map), axis=0),
        )


def process_image_patches(image, model, preprocessing_fn):
    """Process image in patches and reconstruct the full prediction."""
    width, height = image.size

    wi_n = width // PATCH_SIZE
    he_n = height // PATCH_SIZE

    overhang_wi = width - wi_n * PATCH_SIZE
    overhang_he = height - he_n * PATCH_SIZE

    print(f"Overhang (< 1 patch) for width and height: {overhang_wi}, {overhang_he}")

    end_image = None
    end_image_class_map = None

    for h in range(he_n + 1):
        temp_image = None
        temp_image_class_map = None

        for w in range(wi_n + 1):
            # Get patch coordinates and crop image
            crop_coords = get_patch_crop_coordinates(w, h, width, height, wi_n, he_n)
            patch_image = image.crop(crop_coords)

            # Process patch through model
            mask, class_mask = process_single_patch(
                patch_image, model, preprocessing_fn
            )

            # Concatenate horizontally
            temp_image, temp_image_class_map = concatenate_patches_horizontally(
                w, wi_n, overhang_wi, mask, class_mask, temp_image, temp_image_class_map
            )

        # Concatenate vertically
        end_image, end_image_class_map = concatenate_rows_vertically(
            h,
            he_n,
            overhang_he,
            temp_image,
            temp_image_class_map,
            end_image,
            end_image_class_map,
        )

    return end_image, end_image_class_map


def process_single_slide(slide_path: str, model_weights_path: str):
    """Process a single slide for tissue detection.

    Args:
        slide_path (str): Full path to the slide file
        model_weights_path (str): Full path to the model weights file

    Returns:
        numpy.ndarray: Predicted tissue mask, or None if processing failed
    """
    # Get preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, "imagenet")

    # try:
    # Load and initialize model
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        classes=2,
        activation=None,
    )

    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    # slide = OpenSlide(slide_path)
    slide_reader = WSIReader.open(slide_path)

    # Get slide properties
    # width_l0, height_l0 = slide.level_dimensions[0]
    width_l0, height_l0 = slide_reader.slide_dimensions(resolution=0, units="level")
    # mpp = round(float(slide.properties["openslide.mpp-x"]), 4)
    mpp = round(
        float(
            slide_reader.convert_resolution_units(
                input_res=0, input_unit="level", output_unit="mpp"
            )[0]
        ),
        4,
    )
    reduction_factor = DETECTION_MPP / mpp

    # Get thumbnail at target MPP
    # thumbnail_size = (int(width_l0 // reduction_factor),
    #                     int(height_l0 // reduction_factor))
    # image_original = slide.get_thumbnail(thumbnail_size)
    image_original = slide_reader.slide_thumbnail(resolution=DETECTION_MPP, units="mpp")

    # Apply JPEG compression to match training conditions
    compressed_image = apply_jpeg_compression(image_original)
    processed_image = Image.fromarray(compressed_image)

    # Process image patches
    mask_result, class_map_result = process_image_patches(
        processed_image, model, preprocessing_fn
    )
    mask_result = mask_result.astype(np.uint8)
    mask_result[mask_result == 0] = 255  # Set tissue to white
    mask_result[mask_result == 1] = 0  # Set background to black

    del model
    torch.cuda.empty_cache()

    return mask_result

    # except Exception as e:
    #     print(f"Exception processing {slide_path}: {str(e)}")
    #     return None


def main():
    """Main processing function."""
    print("Starting tissue detection analysis...")

    # Initialize setup (only need directory, no longer need model)
    tis_det_dir_mask = os.path.join(OUTPUT_DIR, "tis_det_mask/")
    os.makedirs(tis_det_dir_mask, exist_ok=True)

    slide_names = get_slide_names()

    # Process all slides
    successful_count = 0
    total_slides = len(slide_names)

    for slide_name in slide_names:
        print(f"\nWorking with: {slide_name}")
        slide_path = os.path.join(SLIDE_DIR, slide_name)

        # Process slide and get mask
        mask_result = process_single_slide(slide_path, MODEL_WEIGHT_PATH)

        if mask_result is not None:
            # Convert mask to proper format for saving

            # Save the mask
            mask_filename = f"{slide_name}_MASK.png"
            mask_path = os.path.join(tis_det_dir_mask, mask_filename)
            Image.fromarray(mask_result).save(mask_path)

            print(f"Successfully processed {slide_name}")
            successful_count += 1
        else:
            print(f"Failed to process {slide_name}")

    print(
        f"\nProcessing complete: {successful_count}/{total_slides} slides processed successfully"
    )


if __name__ == "__main__":
    main()
