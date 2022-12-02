import pytest
import torch
import torch.nn.functional as F

from libtilt.inpainting import inpaint


def test_inpaint_2d_tensor():
    """Simple test that the function runs and produces output."""
    image = torch.ones((28, 28))
    mask = F.pad(torch.ones((14, 14)), pad=(7, 7, 7, 7))
    with pytest.warns(UserWarning):
        inpainted = inpaint(
            image,
            mask,
            background_intensity_model_resolution=(5, 5),
            background_intensity_model_samples=200,
        )
    assert inpainted.shape == image.shape


def test_inpaint_3d_tensor():
    """Simple test that the function runs and produces output."""
    image = torch.ones((2, 28, 28))
    mask = F.pad(torch.ones((2, 14, 14)), pad=(7, 7, 7, 7))
    with pytest.warns(UserWarning):
        inpainted = inpaint(
            image=image,
            mask=mask,
            background_intensity_model_resolution=(5, 5),
            background_intensity_model_samples=200,
        )
    assert inpainted.shape == image.shape
