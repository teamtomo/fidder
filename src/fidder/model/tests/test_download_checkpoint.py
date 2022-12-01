from fidder.model import Fidder, download_latest_checkpoint


def test_download_and_load_latest_checkpoint():
    checkpoint_file = download_latest_checkpoint()
    model = Fidder()
    model.load_from_checkpoint(checkpoint_file)
    assert isinstance(model, Fidder)
