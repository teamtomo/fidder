from typing import Tuple, Optional

import numpy as np
import torch
from scipy.interpolate import LSQBivariateSpline
from torch_cubic_spline_grids.b_spline_grids import CubicBSplineGrid3d


def estimate_local_mean(
    image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    resolution: Tuple[int, int] = (5, 5),
    n_samples_for_fit: int = 20000,
):
    """Estimate local mean of an image with a bivariate cubic spline.

    A mask can be provided to

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` array containing image data.
    mask: Optional[torch.Tensor]
        `(h, w)` array containing a binary mask specifying foreground
        and background pixels for the estimation.
    resolution: Tuple[int, int]
        Resolution of the local mean estimate in each dimension.
    n_samples_for_fit: int
        Number of samples taken from foreground pixels for background mean estimation.
        The number of background pixels will be used if this number is greater than the
        number of background pixels.

    Returns
    -------
    local_mean: torch.Tensor
        `(h, w)` array containing a local estimate of the local mean.
    """
    input_dtype = image.dtype
    image = image.numpy()
    mask = np.ones_like(image) if mask is None else mask.numpy()

    # get a random set of foreground pixels for the background fit
    foreground_sample_idx = np.argwhere(mask == 1)

    n_samples_for_fit = min(n_samples_for_fit, len(foreground_sample_idx))
    selection = np.random.choice(
        foreground_sample_idx.shape[0], size=n_samples_for_fit, replace=False
    )
    foreground_sample_idx = foreground_sample_idx[selection]
    y, x = foreground_sample_idx[:, 0], foreground_sample_idx[:, 1]
    z = image[(y, x)]

    # fit a bivariate spline to the data with the specified background model resolution
    ty = np.linspace(0, image.shape[0], num=resolution[0])
    tx = np.linspace(0, image.shape[1], num=resolution[1])
    background_model = LSQBivariateSpline(y, x, z, tx, ty)

    # evaluate the model over a grid covering the whole image
    y = np.arange(image.shape[-2])
    x = np.arange(image.shape[-1])
    local_mean = background_model(y, x, grid=True)
    return torch.tensor(local_mean, dtype=input_dtype)


def estimate_local_mean_3d(
    volume: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    resolution: Tuple[int, int, int] = (5, 5, 5),
    n_samples_for_fit: int = 20000,
):
    """Estimate local mean of an image with a bivariate cubic spline.

    A mask can be provided to

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` array containing image data.
    mask: Optional[torch.Tensor]
        `(h, w)` array containing a binary mask specifying foreground
        and background pixels for the estimation.
    resolution: Tuple[int, int]
        Resolution of the local mean estimate in each dimension.
    n_samples_for_fit: int
        Number of samples taken from foreground pixels for background mean estimation.
        The number of background pixels will be used if this number is greater than the
        number of background pixels.

    Returns
    -------
    local_mean: torch.Tensor
        `(h, w)` array containing a local estimate of the local mean.
    """
    input_dtype = volume.dtype
    volume = volume.numpy()
    mask = np.ones_like(volume) if mask is None else mask.numpy()

    # get a random set of foreground pixels for the background fit
    foreground_sample_idx = np.argwhere(mask == 1)

    n_samples_for_fit = min(n_samples_for_fit, len(foreground_sample_idx))
    selection = np.random.choice(
        foreground_sample_idx.shape[0], size=n_samples_for_fit, replace=False
    )
    foreground_sample_idx = foreground_sample_idx[selection]
    z, y, x = foreground_sample_idx[:, 0], foreground_sample_idx[:, 1], foreground_sample_idx[:, 2]

    w = torch.as_tensor(volume[(z, y, x)])

    grid = CubicBSplineGrid3d(resolution=resolution)
    optimiser = torch.optim.Adam(grid.parameters(), lr=0.01)

    foreground_sample_idx_rescaled = foreground_sample_idx / volume.shape
    for i in range(500):
        # what does the model predict for our observations?
        prediction = grid(foreground_sample_idx_rescaled).squeeze()

        # zero gradients and calculate loss between observations and model prediction
        optimiser.zero_grad()
        loss = torch.sum((prediction - w)**2)**0.5

        # backpropagate loss and update values at points on grid
        loss.backward()
        optimiser.step()

    tz = torch.tensor(np.linspace(0, 1, volume.shape[0]))
    ty = torch.tensor(np.linspace(0, 1, volume.shape[1]))
    tx = torch.tensor(np.linspace(0, 1, volume.shape[2]))
    zz, yy, xx = torch.meshgrid(tz, ty, tx, indexing='xy')
    w = grid(torch.stack((zz, yy, xx), dim=-1)).detach().numpy().reshape(volume.shape)
    return torch.tensor(w, dtype=input_dtype)
