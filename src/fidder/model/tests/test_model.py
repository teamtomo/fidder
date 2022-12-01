import torch
from fidder.model import Fidder


def test_model_instantiation():
    """Instantiation test."""
    model = Fidder()
    assert isinstance(model, Fidder)


def test_model_call():
    """Model should return 2-class logits of same shape."""
    model = Fidder()
    image = torch.rand(size=(10, 1, 512, 512))
    out = model(image)
    assert out.shape == (10, 2, 512, 512)


def test_model_predict_step():
    """Test auto tiled prediction of large single images.

    model.predict_step() should yield probabilities of same shape for class 1.
    """
    model = Fidder()
    image = torch.rand(size=(1024, 1024))
    out = model.predict_step(image)
    assert out.shape == image.shape
    assert torch.min(out) >= 0
    assert torch.max(out) <= 1
