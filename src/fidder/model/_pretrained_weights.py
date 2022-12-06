from pathlib import Path

import pooch

# https://github.com/fatiando/pooch
GOODBOY = pooch.create(
    path=pooch.os_cache("fidder"),
    base_url="doi:10.5281/zenodo.7406503/",
    registry={"fidder.ckpt": "md5:9df1775c157c5b368742e2dbc33f5b10"},
)


def get_latest_checkpoint() -> Path:
    """Retrieve the latest checkpoint from cache if available or download."""
    checkpoint_file = Path(GOODBOY.fetch("fidder.ckpt"))
    return checkpoint_file
