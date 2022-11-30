"""Detect and erase gold fiducials in cryo-EM images."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fidder")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

__all__ = [
    "__version__",
    "cli",
    "Fidder",
    "train_fidder",
]

from ._cli import cli
from .model import Fidder, train_fidder
