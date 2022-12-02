#!/usr/bin/env bash

# predict fiducial mask
fidder predict \
--input-image example.mrc \
--output-mask example_mask.mrc \

# erase masked region
fidder erase \
--input-image example.mrc \
--input-mask example_mask.mrc \
--output-image example_erased.mrc \
