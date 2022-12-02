import torch
from fidder.model import Fidder


def test_model_instantiation():
    """Instantiation test."""
    model = Fidder()
    assert isinstance(model, Fidder)


def test_model_call():
    """Model should return 2-class logits of same shape."""
    model = Fidder()
    image = torch.rand(size=(2, 1, 128, 128))
    out = model(image)
    assert out.shape == (2, 2, 128, 128)


def test_model_predict_step():
    """Test auto tiled prediction of larger single images.

    model.predict_step() should yield probabilities of same shape for class 1.
    """
    model = Fidder(batch_size=1)
    image = torch.rand(size=(512, 512))
    out = model.predict_step(image)
    assert out.shape == image.shape
    assert torch.min(out) >= 0
    assert torch.max(out) <= 1
