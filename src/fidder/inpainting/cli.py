from pathlib import Path

import mrcfile
import numpy as np
import torch
from typer import Option

from .inpainting import inpaint
from ..utils import get_pixel_spacing_from_header
from .._cli import cli, OPTION_PROMPT_KWARGS as PKWARGS


@cli.command(name='erase', no_args_is_help=True)
def erase_segmented_fiducials(
    input_image: Path = Option(..., help='image file in MRC format', **PKWARGS),
    input_mask: Path = Option(..., help='mask file in MRC format', **PKWARGS),
    output: Path = Option(..., help='output file', **PKWARGS),
):
    """Erase a masked region of a cryo-EM image."""
    image = torch.as_tensor(mrcfile.read(input_image))
    mask = torch.as_tensor(mrcfile.read(input_mask), dtype=torch.bool)
    inpainted_image = inpaint(
        image=image,
        mask=mask,
        background_intensity_model_resolution=(8, 8),
        background_intensity_model_samples=25000,
    )
    pixel_spacing = get_pixel_spacing_from_header(input_image)
    mrcfile.write(
        name=output,
        data=np.array(inpainted_image, dtype=np.float32),
        voxel_size=pixel_spacing,
        overwrite=True,
    )
