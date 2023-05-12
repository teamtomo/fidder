from fidder.model import Fidder, get_latest_checkpoint


def test_download_and_load_latest_checkpoint():
    checkpoint_file = get_latest_checkpoint()
    model = Fidder()
    model.load_from_checkpoint(checkpoint_file, map_location="cpu")
    assert isinstance(model, Fidder)
