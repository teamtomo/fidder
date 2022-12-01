from pathlib import Path

import pooch

# https://github.com/fatiando/pooch
GOODBOY = pooch.create(
    path=pooch.os_cache("fidder"),
    base_url="doi:10.5281/zenodo.7386349/",
    registry={
        "fidder.ckpt": "md5:17080b15a67c9224ab16c125499967fc"
    }  # todo: update when new DOI active
)


def download_latest_checkpoint() -> Path:
    checkpoint_file = Path(GOODBOY.fetch("fidder.ckpt"))
    return checkpoint_file
