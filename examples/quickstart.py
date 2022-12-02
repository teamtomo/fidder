import mrcfile
import torch
from fidder.predict import predict_fiducial_mask
from fidder.erase import erase_masked_region

# load your image
image = torch.tensor(mrcfile.read('my_image_file.mrc'))

# use a pretrained model to predict a mask
mask, probabilities = predict_fiducial_mask(image, pixel_spacing=1.35)

# erase fiducials
erased_image = erase_masked_region(image=image, mask=mask)