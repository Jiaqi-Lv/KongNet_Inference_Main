import numpy as np
import torch
from skimage.feature import peak_local_max

from inference.data_utils import morphological_post_processing


def binary_det_post_process(
    prob: torch.Tensor,
    threshold: float = 0.5,
    min_distance: int = 5,
):
    if torch.is_tensor(prob):
        prob = prob.cpu().numpy()

    output_mask = np.zeros(shape=prob.shape, dtype=np.uint8)

    coordinates = peak_local_max(
        prob,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=False,
    )
    output_mask[coordinates[:, 0], coordinates[:, 1]] = 1

    return output_mask


def multihead_det_post_process(
    inflamm_prob: torch.Tensor,
    lymph_prob: torch.Tensor,
    mono_prob: torch.Tensor,
    thresholds: list = [0.5, 0.5, 0.5],
    min_distances: list = [5, 5, 5],
):
    if torch.is_tensor(inflamm_prob):
        inflamm_prob = inflamm_prob.detach().cpu().numpy()
    if torch.is_tensor(lymph_prob):
        lymph_prob = lymph_prob.detach().cpu().numpy()
    if torch.is_tensor(mono_prob):
        mono_prob = mono_prob.detach().cpu().numpy()

    inflamm_output_mask = np.zeros(shape=inflamm_prob.shape, dtype=np.uint8)
    lymph_output_mask = np.zeros(shape=lymph_prob.shape, dtype=np.uint8)
    mono_output_mask = np.zeros(shape=mono_prob.shape, dtype=np.uint8)

    inflamm_coordinates = peak_local_max(
        inflamm_prob,
        min_distance=min_distances[0],
        threshold_abs=thresholds[0],
        exclude_border=False,
    )
    inflamm_output_mask[inflamm_coordinates[:, 0], inflamm_coordinates[:, 1]] = 1

    lymph_coordinates = peak_local_max(
        lymph_prob,
        min_distance=min_distances[1],
        threshold_abs=thresholds[1],
        exclude_border=False,
    )
    lymph_output_mask[lymph_coordinates[:, 0], lymph_coordinates[:, 1]] = 1

    mono_coordinates = peak_local_max(
        mono_prob,
        min_distance=min_distances[2],
        threshold_abs=thresholds[2],
        exclude_border=False,
    )
    mono_output_mask[mono_coordinates[:, 0], mono_coordinates[:, 1]] = 1

    return {
        "inflamm_mask": inflamm_output_mask,
        "lymph_mask": lymph_output_mask,
        "mono_mask": mono_output_mask,
    }


def multihead_det_post_process_batch(
    inflamm_prob: torch.Tensor,
    lymph_prob: torch.Tensor,
    mono_prob: torch.Tensor,
    thresholds: list = [0.5, 0.5, 0.5],
    min_distances: list = [5, 5, 5],
):

    inflamm_output_mask = post_process_batch(
        inflamm_prob,
        threshold=thresholds[0],
        min_distance=min_distances[0],
    )
    lymph_output_mask = post_process_batch(
        lymph_prob,
        threshold=thresholds[1],
        min_distance=min_distances[1],
    )
    mono_output_mask = post_process_batch(
        mono_prob,
        threshold=thresholds[2],
        min_distance=min_distances[2],
    )

    return {
        "inflamm_mask": inflamm_output_mask,
        "lymph_mask": lymph_output_mask,
        "mono_mask": mono_output_mask,
    }


def post_process_batch(
    prob: torch.Tensor,
    threshold: 0.5,
    min_distance: 5,
):

    if torch.is_tensor(prob):
        prob = prob.cpu().numpy()

    prob = np.squeeze(prob, axis=1)

    batches = prob.shape[0]
    output_mask = np.zeros(
        shape=(batches, prob.shape[1], prob.shape[2]),
        dtype=np.uint8,
    )

    for i in range(0, batches):
        coordinates = peak_local_max(
            prob[i],
            min_distance=min_distance,
            threshold_abs=threshold,
            exclude_border=False,
        )
        output_mask[i][coordinates[:, 0], coordinates[:, 1]] = 1

    return output_mask


def multihead_seg_post_process(
    inflamm_prob: torch.Tensor,
    lymph_prob: torch.Tensor,
    mono_prob: torch.Tensor,
    contour_prob: torch.Tensor,
    thresholds: list = [0.5, 0.5, 0.5, 0.3],
) -> dict[str, np.ndarray]:
    """
    Args:
        Thresholds: [overall, lymph, mono, contour]
    """
    if torch.is_tensor(inflamm_prob):
        inflamm_prob = inflamm_prob.detach().cpu().numpy()
    if torch.is_tensor(lymph_prob):
        lymph_prob = lymph_prob.detach().cpu().numpy()
    if torch.is_tensor(mono_prob):
        mono_prob = mono_prob.detach().cpu().numpy()
    if torch.is_tensor(contour_prob):
        contour_prob = contour_prob.detach().cpu().numpy()

    contour_pred_binary = (contour_prob > thresholds[3]).astype(np.uint8)

    overall_pred_binary = (inflamm_prob > thresholds[0]).astype(np.uint8)
    lymph_pred_binary = (lymph_prob > thresholds[1]).astype(np.uint8)
    mono_pred_binary = (mono_prob > thresholds[2]).astype(np.uint8)

    overall_pred_binary[contour_pred_binary > 0] = 0
    lymph_pred_binary[contour_pred_binary > 0] = 0
    mono_pred_binary[contour_pred_binary > 0] = 0

    # Post processing
    overall_pred_binary = morphological_post_processing(overall_pred_binary)
    lymph_pred_binary = morphological_post_processing(lymph_pred_binary)
    mono_pred_binary = morphological_post_processing(mono_pred_binary)

    processed_masks = {
        "inflamm_mask": overall_pred_binary,
        "contour_mask": contour_pred_binary,
        "lymph_mask": lymph_pred_binary,
        "mono_mask": mono_pred_binary,
    }
    return processed_masks


def multihead_seg_post_process_v2(
    inflamm_prob: torch.Tensor,
    lymph_prob: torch.Tensor,
    mono_prob: torch.Tensor,
    inflamm_contour_prob: torch.Tensor,
    lymph_contour_prob: torch.Tensor,
    mono_contour_prob: torch.Tensor,
    thresholds: list = [0.5, 0.5, 0.5, 0.3],
) -> dict[str, np.ndarray]:
    """
    Args:
        Thresholds: [overall, lymph, mono, contour]
    """
    if torch.is_tensor(inflamm_prob):
        inflamm_prob = inflamm_prob.detach().cpu().numpy()
    if torch.is_tensor(lymph_prob):
        lymph_prob = lymph_prob.detach().cpu().numpy()
    if torch.is_tensor(mono_prob):
        mono_prob = mono_prob.detach().cpu().numpy()
    if torch.is_tensor(inflamm_contour_prob):
        inflamm_contour_prob = inflamm_contour_prob.detach().cpu().numpy()
    if torch.is_tensor(lymph_contour_prob):
        lymph_contour_prob = lymph_contour_prob.detach().cpu().numpy()
    if torch.is_tensor(mono_contour_prob):
        mono_contour_prob = mono_contour_prob.detach().cpu().numpy()

    inflamm_contour_pred_binary = (inflamm_contour_prob > thresholds[3]).astype(
        np.uint8
    )
    lymph_contour_pred_binary = (lymph_contour_prob > thresholds[3]).astype(np.uint8)
    mono_contour_pred_binary = (mono_contour_prob > thresholds[3]).astype(np.uint8)

    inflamm_pred_binary = (inflamm_prob > thresholds[0]).astype(np.uint8)
    lymph_pred_binary = (lymph_prob > thresholds[1]).astype(np.uint8)
    mono_pred_binary = (mono_prob > thresholds[2]).astype(np.uint8)

    inflamm_pred_binary[inflamm_contour_pred_binary > 0] = 0
    lymph_pred_binary[lymph_contour_pred_binary > 0] = 0
    mono_pred_binary[mono_contour_pred_binary > 0] = 0

    # Post processing
    inflamm_pred_binary = morphological_post_processing(inflamm_pred_binary)
    lymph_pred_binary = morphological_post_processing(lymph_pred_binary)
    mono_pred_binary = morphological_post_processing(mono_pred_binary)

    processed_masks = {
        "inflamm_mask": inflamm_pred_binary,
        "contour_mask": inflamm_contour_pred_binary,
        "lymph_mask": lymph_pred_binary,
        "mono_mask": mono_pred_binary,
    }
    return processed_masks


def multihead_det_post_process_batch_v2(
    inflamm_prob: torch.Tensor,
    lymph_prob: torch.Tensor,
    mono_prob: torch.Tensor,
    inflamm_seg_prob: torch.Tensor,
    lymph_seg_prob: torch.Tensor,
    mono_seg_prob: torch.Tensor,
    thresholds: list = [0.5, 0.5, 0.5],
    min_distances: list = [5, 5, 5],
):

    inflamm_output_mask = post_process_batch_v2(
        inflamm_prob,
        inflamm_seg_prob,
        threshold=thresholds[0],
        min_distance=min_distances[0],
    )
    lymph_output_mask = post_process_batch_v2(
        lymph_prob,
        lymph_seg_prob,
        threshold=thresholds[1],
        min_distance=min_distances[1],
    )
    mono_output_mask = post_process_batch_v2(
        mono_prob,
        mono_seg_prob,
        threshold=thresholds[2],
        min_distance=min_distances[2],
    )

    return {
        "inflamm_mask": inflamm_output_mask,
        "lymph_mask": lymph_output_mask,
        "mono_mask": mono_output_mask,
    }


def post_process_batch_v2(
    prob: torch.Tensor,
    seg_prob: torch.Tensor,
    threshold: 0.5,
    min_distance: 5,
):

    if torch.is_tensor(prob):
        prob = prob.cpu().numpy()
    if torch.is_tensor(seg_prob):
        seg_prob = seg_prob.cpu().numpy()

    prob = np.squeeze(prob, axis=1)
    seg_prob = np.squeeze(seg_prob, axis=1)

    batches = prob.shape[0]
    output_mask = np.zeros(
        shape=(batches, prob.shape[1], prob.shape[2]),
        dtype=np.uint8,
    )

    for i in range(0, batches):
        prob_mask = prob[i]
        seg_prob_mask = seg_prob[i]
        final_prob_mask = prob_mask * 0.7 + seg_prob_mask * 0.3
        final_prob_mask[prob_mask < threshold] = 0
        coordinates = peak_local_max(
            final_prob_mask,
            min_distance=min_distance,
            threshold_abs=threshold,
            exclude_border=False,
        )
        output_mask[i][coordinates[:, 0], coordinates[:, 1]] = 1

    return output_mask
