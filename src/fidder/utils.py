from pathlib import Path
from typing import Optional, Tuple

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndi
from einops import reduce, rearrange
from torchvision.transforms import functional as TF


def calculate_resampling_factor(source: float, target: float) -> float:
    """Calculate the resampling factor for two different sampling rates."""
    return float(source / target)


def rescale_2d_bicubic(
    image: torch.Tensor,
    factor: Optional[float] = None,
    size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Rescale 2D image(s).

    Parameters
    ----------
    image: torch.Tensor
        `(b, c, h, w)` array of image(s).
    factor: float
        factor by which to rescale image(s).

    Returns
    -------
    rescaled_image: torch.Tensor
        `(b, c, h, w)` array of rescaled image(s).
    """
    if factor is not None:
        h, w = image.shape[-2:]
        size = int(factor * min(h, w))
    rescaled_image = TF.resize(
        image, size=size, interpolation=TF.InterpolationMode.BICUBIC
    )
    return rescaled_image


def rescale_2d_nearest(
    image: torch.Tensor,
    factor: Optional[float] = None,
    size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Rescale 2D image(s).

    Parameters
    ----------
    image: torch.Tensor
        `(b, c, h, w)` array of image(s).
    factor: float
        factor by which to rescale image(s).

    Returns
    -------
    rescaled_image: torch.Tensor
        `(b, c, h, w)` array of rescaled image(s).
    """
    if factor is not None:
        h, w = image.shape[-2:]
        size = int(factor * min(h, w))
    rescaled_image = TF.resize(
        image, size=size, interpolation=TF.InterpolationMode.NEAREST
    )
    return rescaled_image


def normalise_2d(image: torch.Tensor) -> torch.Tensor:
    """Normalise 2D image(s) to mean=0 std=1."""
    mean = reduce(image, "... h w -> ... 1 1", reduction="mean")
    std = torch.std(image, dim=[-2, -1])
    std = rearrange(std, "... -> ... 1 1")
    return (image - mean) / std


def central_crop_2d(image: torch.Tensor, percentage: float = 25) -> torch.Tensor:
    """Get a central crop of (a batch of) 2D image(s).

    Parameters
    ----------
    image: torch.Tensor
        `(b, h, w)` or `(h, w)` array of 2D images.
    percentage: float
        percentage of image height and width for cropped region.
    Returns
    -------
    cropped_image: torch.Tensor
        `(b, h, w)` or `(h, w)` array of cropped 2D images.
    """
    h, w = image.shape[-2], image.shape[-1]
    mh, mw = h // 2, w // 2
    dh, dw = int(h * (percentage / 100 / 2)), int(w * (percentage / 100 / 2))
    hf, wf = mh - dh, mw - dw
    hc, wc = mh + dh, mw + dw
    return image[..., hf:hc, wf:wc]


def central_crop_3d(image: torch.Tensor, percentage: float = 25) -> torch.Tensor:
    """Get a central crop of (a batch of) 2D image(s).

    Parameters
    ----------
    image: torch.Tensor
        `(b, h, w)` or `(h, w)` array of 2D images.
    percentage: float
        percentage of image height and width for cropped region.
    Returns
    -------
    cropped_image: torch.Tensor
        `(b, h, w)` or `(h, w)` array of cropped 2D images.
    """
    h, w, d = image.shape[-3], image.shape[-2], image.shape[-1]
    mh, mw, md = h // 2, w // 2, d // 2
    dh, dw, dd = int(h * (percentage / 100 / 2)), int(w * (percentage / 100 / 2)), int(d * (percentage / 100 / 2))
    hf, wf, df = mh - dh, mw - dw, md - dd
    hc, wc, dc = mh + dh, mw + dw, md + dd
    return image[..., hf:hc, wf:wc, df:dc]


def estimate_background_std(image: torch.Tensor, mask: torch.Tensor):
    """Estimate the standard deviation of the background from a central crop.

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` array containing data for which background standard deviation will be estimated.
    mask: torch.Tensor of 0 or 1
        Binary mask separating foreground and background.
    Returns
    -------
    standard_deviation: float
        estimated standard deviation for the background.
    """
    image = central_crop_2d(image, percentage=25).float()
    mask = central_crop_2d(mask, percentage=25)
    return torch.std(image[mask == 0])


def estimate_background_std_3d(image: torch.Tensor, mask: torch.Tensor):
    """Estimate the standard deviation of the background from a central crop.

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` array containing data for which background standard deviation will be estimated.
    mask: torch.Tensor of 0 or 1
        Binary mask separating foreground and background.
    Returns
    -------
    standard_deviation: float
        estimated standard deviation for the background.
    """
    image = central_crop_3d(image, percentage=25).float()
    mask = central_crop_3d(mask, percentage=25)
    image_masked = image.clone()
    image_masked[mask == 1] = np.nan
    return (
        np.nanmean(np.nanstd(image_masked, axis=0)),
        np.nanmean(np.nanstd(image_masked, axis=1)),
        np.nanmean(np.nanstd(image_masked, axis=2)),
    )


def get_pixel_spacing_from_header(image: Path) -> float:
    with mrcfile.open(image, header_only=True, permissive=True) as mrc:
        return float(mrc.voxel_size.x)


def pixel_count_map_2d(mask: torch.Tensor):
    """Calculate a pixel count map from a binary 2D image.

    https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/60_data_visualization/parametric_maps.html

    Parameters
    ----------
    mask: torch.Tensor
        `(h, w)` binary array

    Returns
    -------

    """
    labels, n = ndi.label(mask.cpu().numpy())
    labels = torch.tensor(labels, dtype=torch.long)
    if n < 100:  # vectorised, not memory efficient
        labels_one_hot = F.one_hot(labels, num_classes=(n + 1))
        counts = reduce(labels_one_hot, "h w c -> c", reduction="sum")
        counts = {label_index: count.item() for label_index, count in enumerate(counts)}
        connected_component_image = np.vectorize(counts.__getitem__)(labels)
        connected_component_image = torch.tensor(connected_component_image)
    else:
        connected_component_image = torch.zeros_like(labels)
        for label_id in range(n + 1):
            conected_component_mask = labels == label_id
            n_connected_components = torch.sum(conected_component_mask)
            connected_component_image[conected_component_mask] = n_connected_components
    return connected_component_image
