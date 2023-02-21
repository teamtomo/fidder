from pathlib import Path

import pooch

# https://github.com/fatiando/pooch
GOODBOY = pooch.create(
    path=pooch.os_cache("fidder"),
    base_url="doi:10.5281/zenodo.7660754/",
    registry={"fidder.ckpt": "md5:6e0eb9d3ed49a7a6a2ad65952e127ced"},
)


def get_latest_checkpoint() -> Path:
    """Retrieve the latest checkpoint from cache if available or download."""
    checkpoint_file = Path(GOODBOY.fetch("fidder.ckpt"))
    return checkpoint_file
