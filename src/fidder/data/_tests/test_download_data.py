from fidder.data import download_training_data


def test_download_training_data(tmp_path):
    download_training_data(tmp_path)
    assert (tmp_path / 'images').exists()
    assert (tmp_path / 'masks').exists()
