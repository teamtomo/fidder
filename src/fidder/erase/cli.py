from pathlib import Path

import einops
import mrcfile
import numpy as np
import torch
from typer import Option

from .erase import erase_masked_region
from ..utils import get_pixel_spacing_from_header
from .._cli import cli, OPTION_PROMPT_KWARGS as PKWARGS


@cli.command(name="erase", no_args_is_help=True)
def erase_masked_region(
    input_image: Path = Option(
        default=...,
        help="Image file in MRC format.",
        **PKWARGS
    ),
    input_mask: Path = Option(
        default=...,
        help="Mask file in MRC format.",
        **PKWARGS
    ),
    output_image: Path = Option(
        default=...,
        help="Output file in MRC format.",
        **PKWARGS
    ),
):
    """Erase a masked region in a cryo-EM image."""
    images = torch.as_tensor(mrcfile.read(input_image)).squeeze().float()
    masks = torch.as_tensor(mrcfile.read(input_mask), dtype=torch.bool).squeeze()
    if images.shape != masks.shape:
        raise ValueError('Shape mismatch between data in image and mask files.')
    images, ps = einops.pack(images, pattern='* h w')
    masks, ps = einops.pack(masks, pattern='* h w')

    erased_images = torch.empty_like(images, dtype=torch.float32)
    for idx, (image, mask) in enumerate(zip(images, masks)):
        _erased_image = erase_masked_region(
            image=image,
            mask=mask,
            background_intensity_model_resolution=(8, 8),
            background_intensity_model_samples=25000,
        )
        erased_images[idx] = _erased_image
    [erased_images] = einops.unpack(erased_images, pattern='* h w', packed_shapes=ps)
    pixel_spacing = get_pixel_spacing_from_header(input_image)
    mrcfile.write(
        name=output_image,
        data=np.array(erased_images, dtype=np.float32),
        voxel_size=pixel_spacing,
        overwrite=True,
    )
