import argparse
import math
import numpy as np
import pickle
import spectral as sp
import spectral.io.envi as envi
import zipfile as zf
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.metrics import accuracy_score
import time
from tqdm import tqdm

TRAIN_FILE_REGISTRY = Path('./hytexila/classification/train.txt')
TEST_FILE_REGISTRY = Path('./hytexila/classification/test.txt')
CLASSIFIER_PKL = Path('hist_classifier.pkl')
DOWNSAMPLED_CHANNELS = 10 # From HyTexila paper
LBP_NEIGHBORS = 8
LBP_RADIUS = 1

def get_parent_dir(img_path):
    """Return the parent directory of the image"""
    prefix_to_dir = {
        'food': Path('./hytexila/ENVI/extracted/food'),
        'stone': Path('./hytexila/ENVI/extracted/stone'),
        'textile': Path('./hytexila/ENVI/extracted/textile'),
        'vegetation': Path('./hytexila/ENVI/extracted/vegetation'),
        'wood': Path('./hytexila/ENVI/extracted/wood')
    }
    img_path_prefix = img_path.split('_')[0]
    if img_path_prefix not in prefix_to_dir:
        raise ValueError("Invalid image path")
    return prefix_to_dir[img_path_prefix]

def get_subimage(img_file, x0, y0, x1, y1):
    """Return a subimage based on the passed image file path and the coordinates."""
    parent_dir = get_parent_dir(img_file)
    img = envi.open(parent_dir / img_file.replace(".raw", ".hdr"), parent_dir / img_file)
    return img[x0:x1+1, y0:y1+1, :] # x1 and y1 are inclusive

def get_subimg_class(img_file):
    """Return the class of the subimage based on the passed image file path."""
    return img_file.replace(".raw", "")

def downsample_channels(img, num_channels):
    '''Downsample channels in image by uniformly sampling num_channels indexes from the full 186 channels'''
    total_channels = img.shape[2]
    keep_every = math.ceil((total_channels-1)/(num_channels+1))
    keep_channels = [(l+1)*keep_every for l in range(num_channels)]
    ds_img = img[:,:,keep_channels]
    # print(keep_channels)
    # print(ds_img.shape)
    # print(img.metadata)
    return ds_img

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

class HistogramClassifier:
    def __init__(self):
        self.model_list = []
        self.model_classes = []

    def add_model_histogram(self, model, class_label):
        self.model_list.append(model)
        self.model_classes.append(class_label)
    
    def predict_class(self, hist):
        """Classify the passed histogram using histogram intersection"""
        max_similarity = 0
        max_class = None
        for i in range(len(self.model_list)):
            similarity = histogram_intersection(hist, self.model_list[i])
            if similarity > max_similarity:
                max_similarity = similarity
                max_class = self.model_classes[i]
        return max_class


def build_histogram_classifier(registry_file, pkl_file, dump_pkl=True):
    hist_classifier = HistogramClassifier()
    train_registry = open(registry_file, 'r')
    lines = train_registry.readlines()
    for line in tqdm(lines):
        if not line:
            break
        subimg_data = line.split('\t')
        if len(subimg_data) != 6:
            continue
        
        subimg_class = get_subimg_class(subimg_data[0])
        subimg = get_subimage(subimg_data[0], int(subimg_data[1]), int(subimg_data[2]), int(subimg_data[3]), int(subimg_data[4]))
        ds_subimg = downsample_channels(subimg, DOWNSAMPLED_CHANNELS)
        hist = compute_lbp(ds_subimg, LBP_NEIGHBORS, LBP_RADIUS)
        hist_classifier.add_model_histogram(hist, subimg_class)
    
    if dump_pkl:
        pickle.dump(hist_classifier, open(pkl_file, 'wb'))
    
    return hist_classifier

if __name__ == "__main__":
    # Algo: For each train image, compute LBP features and find the nearest neighbor in the test set
    # Similarity measure: LBP feature histogram intersection
    # 1. Load all test images
    # 2. Compute LBP histograms for each test image + store in searchable data structure
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--load-classifier', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.load_classifier:
        hist_classifier = pickle.load(open(CLASSIFIER_PKL, 'rb'))
    else:
        # Populate histogram classifier with LBP histograms for each training subimage
        start = time.time()
        hist_classifier = build_histogram_classifier(TRAIN_FILE_REGISTRY, CLASSIFIER_PKL)
        print(f'Building hist classifier took: {time.time() - start}s')

    # Predict classes for each test subimage
    print("Starting test phase...")
    start = time.time()
    test_registry = open(TEST_FILE_REGISTRY, 'r')
    lines = test_registry.readlines()
    preds = []
    targets = []
    for line in tqdm(lines):
        if not line:
            break
        subimg_data = line.split('\t')
        if len(subimg_data) != 6:
            continue

        subimg = get_subimage(subimg_data[0], int(subimg_data[1]), int(subimg_data[2]), int(subimg_data[3]), int(subimg_data[4]))
        ds_subimg = downsample_channels(subimg, DOWNSAMPLED_CHANNELS)
        hist = compute_lbp(ds_subimg, LBP_NEIGHBORS, LBP_RADIUS)
        predicted_class = hist_classifier.predict_class(hist)
        true_class = get_subimg_class(subimg_data[0])
        print(f'predicted class: {predicted_class}, true class: {true_class}')
        preds.append(predicted_class)
        targets.append(true_class)
    
    print(f'accuracy: {accuracy_score(targets, preds)}')
    print(f'Test phase took: {time.time() - start}s')

    # TODO: 
    #  - [X] Write script to extract all zip files
    #  - [X] Figure out how to parse train and test sub-images
    #  - [X] Compute histograms for all test images and store in searchable data structure
    #  - [ ] For each train image, find the nearest neighbor in the test set and compute accuracy over all "train" images