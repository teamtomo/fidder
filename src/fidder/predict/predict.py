from pathlib import Path
from typing import Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from pytorch_lightning import Trainer

from ..model import Fidder, get_latest_checkpoint
from ..utils import calculate_resampling_factor, rescale_2d_bicubic, rescale_2d_nearest
from ..constants import TRAINING_PIXEL_SIZE, PIXELS_PER_FIDUCIAL
from .probabilities_to_mask import probabilities_to_mask


def predict_fiducial_mask(
    image: torch.Tensor,
    pixel_spacing: float,
    probability_threshold: float,
    model_checkpoint_file: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict fiducial masks for a batch of arbitrarily sized images.

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` array containing image data.
    pixel_spacing: float
        Isotropic pixel spacing in Angstroms per pixel.

    Returns
    -------
    mask, probabilities: Tuple[torch.Tensor, torch.Tensor]
        `(h, w)` arrays containing the probability of each pixel belonging to a fiducial.
    """
    # prepare image
    if image.ndim == 2:
        image = rearrange(image, "h w -> 1 1 h w")
    elif image.ndim == 3:
        image = rearrange(image, "b h w -> b 1 h w")
    image = torch.as_tensor(image, dtype=torch.float)
    _, h, w = TF.get_dimensions(image)
    downscale_factor = calculate_resampling_factor(
        source=pixel_spacing, target=TRAINING_PIXEL_SIZE
    )
    image = rescale_2d_bicubic(image, factor=downscale_factor)
    image = rearrange(image, "1 1 h w -> 1 h w")

    # prepare model
    if model_checkpoint_file is None:
        model_checkpoint_file = get_latest_checkpoint()
    model = Fidder.load_from_checkpoint(model_checkpoint_file)
    model.eval()

    # predict
    [probabilities] = Trainer(accelerator="auto").predict(model, image)
    mask = probabilities_to_mask(
        probabilities=probabilities,
        threshold=probability_threshold,
        connected_pixel_count_threshold=(PIXELS_PER_FIDUCIAL // 4),
    )

    # rescale for output
    probabilities = rearrange(probabilities, "h w -> 1 1 h w")
    probabilities = rescale_2d_bicubic(probabilities, size=(h, w))
    probabilities = torch.clamp(probabilities, min=0, max=1)
    rearrange(probabilities, "1 1 h w -> h w")
    mask = rearrange(mask, "h w -> 1 1 h w")
    mask = rescale_2d_nearest(mask, size=(h, w))
    mask = rearrange(mask, "1 1 h w -> h w")
    return mask, probabilities
