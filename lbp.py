import math
import numpy as np
import spectral as sp
import spectral.io.envi as envi
import zipfile as zf
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import feature

# PATH of hyperspectral images
WOOD_DIR = Path('./hytexila/ENVI/wood')
TEXTILE_DIR = Path('./hytexila/ENVI/textile')
SAMPLE = 'wood_01'
DOWNSAMPLED_CHANNELS = 10 # From HyTexila paper
LBP_NEIGHBORS = 8
LBP_RADIUS = 1

def compute_lbp(img, neighbors, radius):
    """Compute LBP for each channel and concatenate into a single feature vector"""
    hist = np.array([])
    for i in range(0, img.shape[2]):
        lbp = feature.local_binary_pattern(img[:,:,i], neighbors, radius, method="default")
        hist_i, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 2**neighbors+1))
        hist = np.concatenate((hist, hist_i))
    return hist

def histogram_intersection(hist_1, hist_2):
    """Nice explanation here: https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html"""
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))  # Normalize
    return intersection

if __name__ == "__main__":
    # Algo: For each train image, compute LBP features and find the nearest neighbor in the test set
    # Similarity measure: LBP feature histogram intersection
    # 1. Load all test images
    # 2. Compute LBP histograms for each test image + store in searchable data structure

    with zf.ZipFile((WOOD_DIR / SAMPLE).with_suffix(".zip")) as archive:
        # print(archive.namelist())
        img = envi.open((WOOD_DIR / SAMPLE).with_suffix(".hdr"), (WOOD_DIR / SAMPLE).with_suffix(".raw"))
    
    # Downsample channels in image by uniformly sampling NUM_CHANNELS indexes from the full 186 channels.
    total_channels = img.shape[2]
    keep_every = math.ceil((total_channels-1)/(DOWNSAMPLED_CHANNELS+1))
    keep_channels = [(l+1)*keep_every for l in range(DOWNSAMPLED_CHANNELS)]
    ds_img = img[:,:,keep_channels]
    print(keep_channels)
    print(ds_img.shape)
    # print(img.metadata)

    # Compute LBP for each channel and concatenate into a single feature vector
    hist = compute_lbp(ds_img, LBP_NEIGHBORS, LBP_RADIUS)
    print(hist.shape)

    img2 = envi.open((WOOD_DIR / 'wood_02').with_suffix(".hdr"), (WOOD_DIR / 'wood_02').with_suffix(".raw"))
    ds_img2 = img2[:,:,keep_channels]
    hist2 = compute_lbp(ds_img2, LBP_NEIGHBORS, LBP_RADIUS)

    hist_similarity = histogram_intersection(hist, hist2)
    print(hist_similarity)

    # TODO: 
    #  - [X] Write script to extract all zip files
    #  - Figure out how to parse train and test sub-images
    #  - Compute histograms for all test images and store in searchable data structure
    #  - For each train image, find the nearest neighbor in the test set and compute accuracy over all "train" images