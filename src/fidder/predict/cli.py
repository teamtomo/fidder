from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np
import torch
from typer import Option

from .predict import predict_fiducial_mask as _predict_fiducial_mask
from ..utils import (
    get_pixel_spacing_from_header,
)
from .._cli import cli, OPTION_PROMPT_KWARGS as PKWARGS


@cli.command(name="predict", no_args_is_help=True)
def predict_fiducial_mask(
    input_image: Path = Option(
        default=..., help="Input image file in MRC format.", **PKWARGS
    ),
    pixel_spacing: Optional[float] = Option(
        default=None, help="Pixel spacing in ångströms."
    ),
    probability_threshold: float = Option(
        default=0.5,
        help="Threshold above which pixels are considered part of a fiducial.",
    ),
    output_mask: Path = Option(
        default=..., help="Output mask file in MRC format.", **PKWARGS
    ),
    output_probabilities: Optional[Path] = Option(
        default=None, help="Output probability image file in MRC format."
    ),
    model_checkpoint_file: Optional[Path] = Option(
        default=None, help="File containing segmentation model checkpoint."
    ),
):
    """Predict a fiducial mask using a pretrained model."""
    image = torch.tensor(mrcfile.read(input_image))
    if pixel_spacing is None:
        pixel_spacing = get_pixel_spacing_from_header(input_image)
    mask, probabilities = _predict_fiducial_mask(
        image=image,
        pixel_spacing=pixel_spacing,
        probability_threshold=probability_threshold,
        model_checkpoint_file=model_checkpoint_file,
    )
    mask = mask.cpu().numpy().astype(np.int8)
    probabilities = probabilities.cpu().numpy()
    output_pixel_spacing = (1, pixel_spacing, pixel_spacing)
    mrcfile.write(
        name=output_mask, data=mask, voxel_size=output_pixel_spacing, overwrite=True
    )
    if output_probabilities is not None:
        probabilities = probabilities.astype(np.float32)
        mrcfile.write(
            name=output_probabilities,
            data=probabilities,
            voxel_size=output_pixel_spacing,
            overwrite=True,
        )
