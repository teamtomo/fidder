from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from pytorch_lightning import Trainer

from ..model import Fidder, get_latest_checkpoint
from ..utils import calculate_resampling_factor, rescale_2d_bicubic
from ..constants import TRAINING_PIXEL_SIZE


def predict_fiducial_probabilities(
    image: torch.Tensor,
    pixel_spacing: float,
    checkpoint_file: Optional[Path] = None
) -> np.ndarray:
    """Predict fiducial masks for a batch of arbitrarily sized images.


    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` or `(b, h, w)` array containing image data.
    pixel_spacing: float
        Isotropic pixel spacing in Angstroms per pixel.

    Returns
    -------
    probabilities: torch.Tensor
        `(b, h, w)` array containing the probability of each pixel belonging to a fiducial.
    """
    if image.ndim == 2:
        image = rearrange(image, 'h w -> 1 1 h w')
    elif image.ndim == 3:
        image = rearrange(image, 'b h w -> b 1 h w')
    image = torch.as_tensor(image, dtype=torch.float)
    _, h, w = TF.get_dimensions(image)
    downscale_factor = calculate_resampling_factor(
        source=pixel_spacing, target=TRAINING_PIXEL_SIZE
    )
    image = rescale_2d_bicubic(image, factor=downscale_factor)

    if checkpoint_file is None:
        checkpoint_file = get_latest_checkpoint()
    model = Fidder.load_from_checkpoint(checkpoint_file)
    model.eval()
    image = rearrange(image, '1 1 h w -> 1 h w')
    [probabilities] = Trainer(accelerator="auto").predict(model, image)
    probabilities = rearrange(probabilities, 'h w -> 1 1 h w')
    probabilities = rescale_2d_bicubic(probabilities, size=(h, w))
    probabilities = torch.clamp(probabilities, min=0, max=1)
    return rearrange(probabilities, 'b 1 h w -> b h w')
