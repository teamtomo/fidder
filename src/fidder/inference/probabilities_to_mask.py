import torch
from ..utils import connected_component_transform_2d


def probabilities_to_mask(
    probabilities: torch.Tensor,
    threshold: float = 0.5,
    connected_pixel_count_threshold: int = 20,
) -> torch.Tensor:
    """Derive a mask from per-pixel probabilities.

    Threshold probabilities and remove small disconnected regions.

    Parameters
    ----------
    probabilities: torch.Tensor
        `(h, w)` array of per-pixel probabilities.
    threshold: float
        Threshold value for masking.
    connected_pixel_count_threshold: int
        Minimum number of connected pixels for a region to be considered part of
        the final mask.

    Returns
    -------
    mask: torch.Tensor
        `(h, w)` boolean array.
    """
    mask = probabilities > threshold
    pixel_count_map = connected_component_transform_2d(mask)
    mask[pixel_count_map < connected_pixel_count_threshold] = 0
    return mask
