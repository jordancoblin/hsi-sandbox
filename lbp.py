import math
import numpy as np
import spectral as sp
import spectral.io.envi as envi
import zipfile as zf
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import feature

# PATH of hyperspectral images
DATA_DIR = Path('./hytexila/ENVI/wood')
SAMPLE = 'wood_01'
DOWNSAMPLED_CHANNELS = 10 # From HyTexila paper
LBP_NEIGHBORS = 8
LBP_RADIUS = 1

if __name__ == "__main__":
    # Algo: For each train image, compute LBP features and find the nearest neighbor in the test set
    # Similarity measure: LBP feature histogram intersection
    # 1. Load all test images
    # 2. Compute LBP histograms for each test image + store in searchable data structure

    with zf.ZipFile((DATA_DIR / SAMPLE).with_suffix(".zip")) as archive:
        # print(archive.namelist())
        img = envi.open((DATA_DIR / SAMPLE).with_suffix(".hdr"), (DATA_DIR / SAMPLE).with_suffix(".raw"))
    
    # Downsample channels in image by uniformly sampling NUM_CHANNELS indexes from the full 186 channels.
    total_channels = img.shape[2]
    keep_every = math.ceil((total_channels-1)/(DOWNSAMPLED_CHANNELS+1))
    keep_channels = [(l+1)*keep_every for l in range(DOWNSAMPLED_CHANNELS)]
    ds_img = img[:,:,keep_channels]
    print(keep_channels)
    print(ds_img.shape)
    # print(img.metadata)

    # Compute LBP features for each pixel in the image
    lbp = feature.local_binary_pattern(ds_img[:,:,0], LBP_NEIGHBORS, LBP_RADIUS, method="default")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 2**LBP_NEIGHBORS+1))

    # TODO: compute LBP for each channel and concatenate into a single feature vector