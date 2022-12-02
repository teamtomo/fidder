import torch

from fidder.utils import connected_component_transform_2d


def test_connected_component_transform_2d():
    mask = torch.zeros((10, 10))
    mask[::2, ::2] = 1
    mask[1::2, 1::2] = 1
    connected_component_image = connected_component_transform_2d(mask).long()
    assert torch.allclose(
        connected_component_image[::2, ::2], torch.tensor(1).long())
    assert torch.allclose(
        connected_component_image[1::2, 1::2], torch.tensor(1).long()
    )
    assert torch.allclose(
        connected_component_image[1::2, ::2], torch.tensor(50).long()
    )
    assert torch.allclose(
        connected_component_image[::2, 1::2], torch.tensor(50).long()
    )
