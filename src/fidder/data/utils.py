import torch
import torchvision.transforms.functional as TF
from einops import reduce, rearrange


def calculate_resampling_factor(source: float, target: float) -> float:
    """Calculate the resampling factor for two different sampling rates."""
    return float(source / target)


def rescale_2d_bicubic(
    image: torch.Tensor, factor: float
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
    _, h, w = TF.get_dimensions(image)
    short_edge_length = int(factor * min(h, w))
    rescaled_image = TF.resize(
        image,
        size=short_edge_length,
        interpolation=TF.InterpolationMode.BICUBIC
    )
    return rescaled_image


def normalise_2d(image: torch.Tensor) -> torch.Tensor:
    """Normalise 2D image(s) to mean=0 std=1."""
    mean = reduce(image, '... h w -> ... 1 1', reduction='mean')
    std = torch.std(image, dim=[-2, -1])
    std = rearrange(std, '... -> ... 1 1')
    return (image - mean) / std
