import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any, List

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
from huggingface_hub import hf_hub_download
from scipy import ndimage
from scipy.spatial import KDTree
from shapely import Point, Polygon
from skimage.feature import peak_local_max
from skimage.measure import label
from tiatoolbox.annotation.storage import Annotation, AnnotationStore, SQLiteStore
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.wsicore.wsireader import WSIReader
from torch.amp import autocast
from tqdm import tqdm


def open_json_file(json_path: str):
    """Extract annotations from json file"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def imagenet_denormalise(img: np.ndarray) -> np.ndarray:
    """Normalize RGB image to ImageNet mean and std"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std) + mean
    return img


def imagenet_normalise(img: np.ndarray) -> np.ndarray:
    """Revert ImageNet normalized RGB"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img - mean
    img = img / std
    return img


def imagenet_normalise_torch(img: torch.Tensor) -> torch.Tensor:
    """Normalises input image to ImageNet mean and std
    Input torch tensor (B,3,H,W)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
    return (img - mean) / std


def px_to_mm(px: int, mpp: float = 0.24199951445730394):
    """
    Convert pixel coordinate to millimeters
    """
    return px * mpp / 1000


def mm_to_px(mm: float, mpp: float = 0.24199951445730394) -> float:
    """
    Convert millimeter coordinate to pixels
    """
    return mm * 1000 / mpp


def write_json_file(location: str, content: Any) -> None:
    """Write content to JSON file at specified location."""
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def collate_fn(batch: List[Any]) -> torch.Tensor:
    """Custom collate function for DataLoader.

    Args:
        batch (list): List of data samples

    Returns:
        torch.Tensor: Batched tensor data
    """
    # Apply the make_writable function to each element in the batch
    batch = np.asarray(batch)
    writable_batch = batch.copy()
    # Convert each element to a tensor
    return torch.as_tensor(writable_batch, dtype=torch.float)


def check_coord_in_mask(x, y, mask, coord_res, mask_res):
    """Checks if a given coordinate is inside the tissue mask
    Coordinate (x, y)
    Binary tissue mask default at 1.25x
    """
    if mask is None:
        return True

    try:
        return mask[int(np.round(y)), int(np.round(x))] == 1
    except IndexError:
        return False


def scale_coords(coords: list, scale_factor: float = 1):
    new_coords = []
    for coord in coords:
        x = int(coord[0] * scale_factor)
        y = int(coord[1] * scale_factor)
        new_coords.append([x, y])

    return new_coords


def _build_annotation(record: dict, scale_factor: float, shape_type: str) -> Annotation:
    x = record["x"] * scale_factor
    y = record["y"] * scale_factor

    if shape_type == "polygon":
        geometry = Polygon.from_bounds(x - 16, y - 16, x + 16, y + 16)
    else:
        geometry = Point(x, y)

    return Annotation(
        geometry=geometry,
        properties={
            "type": record["type"],
            "prob": record["prob"],
        },
    )


def detection_to_annotation_store(
    detection_records: list[dict],
    scale_factor: float = 1,
    shape_type="polygon",
    max_workers: int = 4,
):
    annotation_store = SQLiteStore()

    print("Building annotations from detection records...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        annotations = list(
            executor.map(
                lambda rec: _build_annotation(rec, scale_factor, shape_type),
                detection_records,
            )
        )
    print("Appending annotations to store...")
    annotation_store.append_many(annotations)
    return annotation_store


def detection_to_csv(
    detection_records: list[dict],
    scale_factor: float = 1,
    csv_path: str = "detections.csv",
):

    df = pd.DataFrame(detection_records)
    df["x"] = df["x"].apply(lambda x: x * scale_factor)
    df["y"] = df["y"].apply(lambda y: y * scale_factor)
    df.to_csv(csv_path, index=False)

    return


def filter_detection_with_mask(
    detection_records: list[dict],
    mask: np.ndarray,
    points_mpp: float = 0.24199951445730394,
    mask_mpp: float = 8.0,
    margin: int = 1,
) -> list[dict]:
    """
    Filter detected points: [{'x','y','type','prob'}]
    Using binary mask.
    Points outside the binary mask are removed

    Args:
        detection_records: [{'x','y','type','prob'}]
        mask: binary mask to for filtering
        points_mpp: resolution of the detected points in mpp
        mask_mpp: resolution of the binary mask in mpp
        margin: margin in pixels to add around the mask
    Returns:
        fitlered_records: [{'x','y','type','prob'}]
    """
    scale_factor = mask_mpp / points_mpp
    print(f"Scale factor: {scale_factor}")

    filtered_records: list[dict] = []
    for record in tqdm(
        detection_records,
        desc="Filtering detections with mask",
        total=len(detection_records),
    ):
        x = record["x"]
        y = record["y"]
        print(record)

        x_in_mask = int(np.round(x / scale_factor))
        y_in_mask = int(np.round(y / scale_factor))
        top_left = (x_in_mask - margin, y_in_mask - margin)
        top_right = (x_in_mask + margin, y_in_mask - margin)
        bottom_left = (x_in_mask - margin, y_in_mask + margin)
        bottom_right = (x_in_mask + margin, y_in_mask + margin)
        indices = [
            (int(round(xi)), int(round(yi)))
            for xi, yi in [
                top_left,
                top_right,
                bottom_left,
                bottom_right,
            ]
        ]
        valid_indices = [
            (xi, yi)
            for xi, yi in indices
            if 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]
        ]
        ones_count = sum(mask[yi, xi] for xi, yi in valid_indices)
        if len(valid_indices) == 0:
            continue
        else:
            if ones_count / len(valid_indices) >= 0.5:
                filtered_records.append(record)
            else:
                continue
        # try:
        #     ones_count = sum(mask[])
        #     if mask[y_in_mask, x_in_mask] != 0:
        #         filtered_records.append(record)
        #     else:
        #         continue
        # except IndexError:
        #     continue

    return filtered_records


def non_max_suppression_fast(boxes, overlapThresh):
    """Very efficient NMS function taken from pyimagesearch"""

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs,
            np.concatenate(([last], np.where(overlap > overlapThresh)[0])),
        )
    # return only the bounding boxes that were picked using the
    # integer data type
    # return boxes[pick].astype("int")
    return pick


def nms(boxes: np.ndarray, overlapThresh: float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        overlapThresh: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # we extract coordinates for every
    # prediction box present in P
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # we extract the confidence scores as well
    scores = boxes[:, 4]

    # calculate area of every block in P
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # sort the prediction boxes in P
    # according to their confidence scores
    idxs = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    pick = []

    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs,
            np.concatenate(([last], np.where(overlap > overlapThresh)[0])),
        )
    # return only the bounding boxes that were picked using the
    # integer data type
    # return boxes[pick].astype("int")

    return pick


def get_centerpoints(box, dist):
    """Returns centerpoints of box"""
    return (box[0] + dist, box[1] + dist)


def get_points_within_box(annotation_store: AnnotationStore, box) -> list:
    query_poly = Polygon.from_bounds(box[0], box[1], box[2], box[3])
    anns = annotation_store.query(geometry=query_poly)
    results = []
    for point in anns.items():
        entry = {
            "x": point[1].coords[0][0],
            "y": point[1].coords[0][1],
            "type": point[1].properties["type"],
            "prob": point[1].properties["prob"],
        }
        results.append(entry)
    return results


def point_to_box(x, y, size, prob=None):
    """
    Convert centerpoint to bounding box of fixed size
    Args:
        x: x coordinate
        y: y coordinate
        size: radius of the box
        prob: probability of the point
    Returns:
        box: np.ndarray[4], if prob is not None [5]
    """
    if prob == None:
        return np.array([x - size, y - size, x + size, y + size])
    else:
        return np.array([x - size, y - size, x + size, y + size, prob])


def _process_single_tile_nms(
    bb, annotation_store_path, tile_patch_size, box_size, overlap_thresh
):
    annotation_store = SQLiteStore.open(annotation_store_path)
    x_pos, y_pos = bb[0], bb[1]
    box = [x_pos, y_pos, x_pos + tile_patch_size[0], y_pos + tile_patch_size[1]]

    patch_points = get_points_within_box(annotation_store, box)

    if len(patch_points) < 2:
        return patch_points

    boxes = np.array(
        [point_to_box(p["x"], p["y"], box_size, p["prob"]) for p in patch_points]
    )
    indices = nms(boxes, overlap_thresh)
    return [patch_points[i] for i in indices]


# After each patch prediction, do peak detection and nms on the patch.
# Then


def slide_nms(
    wsi_reader: WSIReader,
    binary_mask: np.ndarray,
    detection_record: list[dict] | SQLiteStore,
    detection_mpp: float = 0.25,
    tile_size: int = 4007,
    box_size: int = 5,
    overlap_thresh: float = 0.5,
    cache_dir: str = "./",
    num_workers: int = 10,
) -> list[dict]:
    tile_patch_size = [tile_size + 77, tile_size]
    tile_extractor = get_patch_extractor(
        input_img=wsi_reader,
        input_mask=binary_mask,
        method_name="slidingwindow",
        patch_size=tile_patch_size,
        resolution=detection_mpp,
        units="mpp",
    )

    if isinstance(detection_record, SQLiteStore):
        annotation_store = detection_record
    else:
        annotation_store = detection_to_annotation_store(
            detection_record, scale_factor=1, shape_type="Point"
        )

    annotation_store_path = os.path.join(cache_dir, "temp_store_1.db")
    annotation_store.dump(annotation_store_path)

    tile_coords = tile_extractor.coordinate_list
    temp_nms_points = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _process_single_tile_nms,
                bb,
                annotation_store_path,
                tile_patch_size,
                box_size,
                overlap_thresh,
            )
            for bb in tile_coords
        ]

        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Multiprocess NMS stage 1"
        ):
            result = f.result()
            temp_nms_points.extend(result)

    print(f"Number of points after first NMS stage: {len(temp_nms_points)}")

    os.remove(annotation_store_path)

    # Do it again with a bigger tile size
    annotation_store = detection_to_annotation_store(
        temp_nms_points, scale_factor=1, shape_type="Point"
    )
    annotation_store_path = os.path.join(cache_dir, "temp_store_2.db")
    annotation_store.dump(annotation_store_path)

    tile_patch_size = [tile_size, tile_size + 77]
    tile_extractor = get_patch_extractor(
        input_img=wsi_reader,
        input_mask=binary_mask,
        method_name="slidingwindow",
        patch_size=tile_patch_size,
        resolution=detection_mpp,
        units="mpp",
    )

    tile_coords = tile_extractor.coordinate_list
    final_nms_points = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _process_single_tile_nms,
                bb,
                annotation_store_path,
                tile_patch_size,
                box_size,
                overlap_thresh,
            )
            for bb in tile_coords
        ]

        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Multiprocess NMS stage 2"
        ):
            result = f.result()
            final_nms_points.extend(result)

    os.remove(annotation_store_path)

    print(f"Number of points after second NMS stage: {len(final_nms_points)}")

    return final_nms_points


def erode_mask(mask, size=3, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    if mask.ndim == 4:
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask[i, j, :, :] = cv2.erode(
                    mask[i, j, :, :], kernel, iterations=iterations
                )
    else:
        mask = cv2.erode(mask, kernel, iterations=iterations)

    return mask


def morphological_post_processing(mask, size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    if mask.ndim == 4:
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask[i, j, :, :] = cv2.morphologyEx(
                    mask[i, j, :, :], cv2.MORPH_OPEN, kernel
                )
                mask[i, j, :, :] = cv2.morphologyEx(
                    mask[i, j, :, :], cv2.MORPH_CLOSE, kernel
                )
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def check_image_mask_shape(wsi_path: str, mask_path: str) -> None:
    """
    Check if the image and mask have the same shape and mpp
    """
    wsi_reader = WSIReader.open(wsi_path)
    wsi_shape = wsi_reader.slide_dimensions(resolution=0, units="level")
    mask_reader = WSIReader.open(mask_path)
    mask_shape = mask_reader.slide_dimensions(resolution=0, units="level")

    if (wsi_shape[0] != mask_shape[0]) or (wsi_shape[1] != mask_shape[1]):
        message = f"Image and mask have different shapes: {wsi_shape} vs {mask_shape}"
        raise ValueError(message)

    wsi_info = wsi_reader.info.as_dict()
    mask_info = mask_reader.info.as_dict()
    wsi_mpp = wsi_info["mpp"]
    mask_mpp = mask_info["mpp"]
    if (round(wsi_mpp[0], 3) != round(mask_mpp[0], 3)) or (
        round(wsi_mpp[1], 3) != round(mask_mpp[1], 3)
    ):
        message = f"Image and mask have different mpp: {wsi_mpp} vs {mask_mpp}"
        raise ValueError(message)


def download_weights_from_hf(
    checkpoint_name: str, repo_id: str, save_dir: str, force_download: bool = False
) -> str:
    """
    Download model weights from Hugging Face
    """
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_name,
        local_dir=save_dir,
        force_download=force_download,
    )
    return checkpoint_path
