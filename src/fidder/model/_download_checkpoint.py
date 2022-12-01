from pathlib import Path

import pooch

# https://github.com/fatiando/pooch
GOODBOY = pooch.create(
    path=pooch.os_cache("fidder"),
    base_url="doi:10.5281/zenodo.7386812/",
    registry={
        "fidder.ckpt": "md5:020a01feb2f3f5bc5a2e519e6009f562"
    }
)


def download_latest_checkpoint() -> Path:
    checkpoint_file = Path(GOODBOY.fetch("fidder.ckpt"))
    return checkpoint_file
