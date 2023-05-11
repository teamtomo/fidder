"""Detect and erase gold fiducials in cryo-EM images."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fidder")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

__all__ = ["__version__", "cli", "Fidder", "train_fidder", "download_training_data"]

from ._cli import cli
# cli tools
from .train import train_fidder
from .predict.cli import predict_fiducial_mask
from .erase.cli import erase_masked_region

# python things
from .model import Fidder
from .data import download_training_data
from .predict import predict_fiducial_mask
from .erase import erase_masked_region


