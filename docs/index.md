# Overview

[![License](https://img.shields.io/pypi/l/fidder.svg?color=green)](https://github.com/teamtomo/fidder/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/fidder.svg?color=green)](https://pypi.org/project/fidder)
[![Python Version](https://img.shields.io/pypi/pyversions/fidder.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/fidder/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/fidder/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/fidder/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/fidder)


*fidder* is a Python package for detecting and erasing gold fiducials in cryo-EM
images.

<script
  defer
  src="https://unpkg.com/img-comparison-slider@7/dist/index.js"
></script>


<img-comparison-slider tabindex="0">
  <img slot="first" src="https://user-images.githubusercontent.com/7307488/205206563-00944ef6-02b9-4830-9e67-86daed9ffffb.png"/>
  <img slot="second" src="https://user-images.githubusercontent.com/7307488/205206583-c9df5cdb-2034-484b-99d2-ce07827e90e3.png" />
</img-comparison-slider>

The package can be used from both
[Python](usage/python.md)
and the
[command line](usage/command_line.md).

---

# Quickstart

## Python

```python
import mrcfile
import torch
from fidder.predict import predict_fiducial_mask
from fidder.erase import erase_masked_region

# load your image
image = torch.tensor(mrcfile.read('my_image_file.mrc'))

# use a pretrained model to predict a mask
mask, probabilities = predict_fiducial_mask(
    image, pixel_spacing=1.35, probability_threshold=0.5
)

# erase fiducials
erased_image = erase_masked_region(image=image, mask=mask)
```
## Command Line

```bash
# predict fiducial mask
fidder predict \
--input-image example.mrc \
--probability-threshold 0.5 \
--output-mask mask.mrc

# erase masked region
fidder erase \
--input-image example.mrc \
--input-mask mask.mrc \
--output-image erased.mrc

```

---

# Installation

pip:

```shell
pip install fidder
```