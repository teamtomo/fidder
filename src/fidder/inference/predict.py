import numpy as np
import torch
from einops import rearrange
from pytorch_lightning import Trainer
from rich.console import Console

from ..model import Fidder
from ..data.utils import calculate_resampling_factor, rescale_2d_bicubic
from ..constants import TRAINING_PIXEL_SIZE

console = Console()


def predict_fiducial_probabilities(
        image: torch.Tensor,
        pixel_spacing: float,
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
        rearrange(image, 'h w -> 1 1 h w')
    elif image.ndim == 3:
        rearrange(image, 'b h w -> b 1 h w')
    image = torch.as_tensor(image, dtype=torch.float)
    rescale_factor = calculate_resampling_factor(
        source=pixel_spacing, target=TRAINING_PIXEL_SIZE
    )
    image = rescale_2d_bicubic(image, factor=rescale_factor)

    model = Fidder()
    probabilities = Trainer(accelerator="auto").predict(model, image)
    probabilities = rearrange(probabilities, 'b h w -> b 1 h w')
    probabilities = rescale_2d_bicubic(probabilities, factor=1/rescale_factor)
    probabilities = torch.clamp(probabilities, min=0, max=1)
    return rearrange(probabilities, 'b 1 h w -> b h w')


