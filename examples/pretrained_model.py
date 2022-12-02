from pathlib import Path

from fidder.predict.cli import predict_fiducial_mask
from fidder.erase.cli import erase_segmented_fiducials

image = Path('TS_01_0deg_bin8.mrc')
mask = Path(image.stem + '_mask.mrc')
probabilities = Path(image.stem + '_probabilities.mrc')
checkpoint = Path(
    '../training/lightning_logs/version_3123283/checkpoints/epoch=24-step=600.ckpt')
erased = Path(image.stem + '_erased.mrc')

predict_fiducial_mask(
    input_image=image,
    pixel_spacing=None,
    probability_threshold=0.5,
    output_mask=mask,
    output_probabilities=probabilities,
    model_checkpoint_file=checkpoint
)

erase_segmented_fiducials(
    input_image=image,
    input_mask=mask,
    output=erased,
)
