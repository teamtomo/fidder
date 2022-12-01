from pathlib import Path
from typing import Optional

import einops
import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
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
    checkpoint_file: Optional[Path] = None
):
    image = torch.tensor(mrcfile.read(input_image))
    if pixel_spacing is None:
        pixel_spacing = get_pixel_spacing_from_header(input_image)
    image = rearrange(image, 'h w -> 1 h w')
    probabilities = predict_fiducial_probabilities(
        image=image,
        pixel_spacing=pixel_spacing,
        checkpoint_file=checkpoint_file,
    )
    probabilities = probabilities.cpu().numpy()
    probabilities = rearrange(probabilities, '1 h w -> h w')
    mask = probabilities > 0.2
    mask = mask.astype(np.int8)
    output_pixel_spacing = (1, pixel_spacing, pixel_spacing)
    mrcfile.write(
        name=output_mask,
        data=mask,
        voxel_size=output_pixel_spacing,
        overwrite=True
    )
    if output_probabilities is not None:
        probabilities = probabilities.astype(np.float32)
        mrcfile.write(
            name=output_probabilities,
            data=probabilities,
            voxel_size=output_pixel_spacing,
            overwrite=True
        )
