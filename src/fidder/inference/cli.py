from pathlib import Path
from typing import Optional

import einops
import mrcfile
import torch
import torch.nn.functional as F
from einops import rearrange
from typer import Option
from scipy import ndimage as ndi

from ..constants import TRAINING_PIXEL_SIZE
from .predict import predict_fiducial_probabilities
from ..utils import (
    get_pixel_spacing_from_header,
    calculate_resampling_factor,
    rescale_2d_bicubic,
    rescale_2d_nearest,
)
from .._cli import cli, OPTION_PROMPT_KWARGS as PKWARGS


@cli.command(name='predict', no_args_is_help=True)
def predict_fiducial_mask(
    input_image: Path = Option(
        default=..., help='input image file in MRC format', **PKWARGS
    ),
    pixel_spacing: Optional[float] = Option(
        default=None, help='pixel spacing in ångströms'
    ),
    output_mask: Path = Option(
        default=..., help='output mask file in MRC format', **PKWARGS
    ),
    output_probabilities: Optional[Path] = Option(
        default=None, help='output probability image in ångströms'
    ),
):
    image = torch.tensor(mrcfile.read(input_image))
    if pixel_spacing is None:
        pixel_spacing = get_pixel_spacing_from_header(input_image)
    rescale_factor = calculate_resampling_factor(
        source=pixel_spacing, target=TRAINING_PIXEL_SIZE
    )
    image = rearrange(image, 'h w -> 1 1 h w')
    image = rescale_2d_bicubic(image, factor=rescale_factor)
    image = rearrange(image, 'h w -> h w')
    probabilities = predict_fiducial_probabilities(image, pixel_spacing)
    mask = rearrange(probabilities > 0.2, '1 h w -> h w')



