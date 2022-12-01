from pathlib import Path
from typing import Optional

import mrcfile
import torch
from typer import Option

from .predict import predict_fiducial_probabilities
from ..utils import get_pixel_spacing_from_header
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
    probabilities = predict_fiducial_probabilities(image, pixel_spacing)
    mask
