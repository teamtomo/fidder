from pathlib import Path

from fidder.inference.cli import predict_fiducial_mask
from fidder.inpainting.cli import erase_segmented_fiducials

image = Path('Pos10_ts_001_0001_-0_0.mrc')
mask = Path(image.stem + '_mask.mrc')
probabilities = Path(image.stem + '_probabilities.mrc')
checkpoint = Path('../training/lightning_logs/version_3123283/checkpoints/epoch=24-step=600.ckpt')
erased = Path(image.stem + '_erased.mrc')

predict_fiducial_mask(
    input_image=image,
    pixel_spacing=None,
    output_mask=mask,
    output_probabilities=probabilities,
    checkpoint_file=checkpoint
)

erase_segmented_fiducials(
    input_image=image,
    input_mask=mask,
    output=erased,
)
