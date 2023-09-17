import math
import numpy as np
import os
import spectral as sp
import spectral.io.envi as envi
import zipfile as zf
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import feature

ZIP_DIR = Path('./hytexila/ENVI/compressed')

def extract_zip_file(root, file):
    """Extracts a zip file to the target directory"""
    with zf.ZipFile(os.path.join(root, file)) as archive:
        target_dir = root.replace("compressed", "extracted")
        archive.extract(file.replace(".zip", ".hdr"), target_dir)
        archive.extract(file.replace(".zip", ".raw"), target_dir)

if __name__ == "__main__":
    for root, subdirs, files in os.walk(ZIP_DIR):
        for file in files:
            if file.endswith(".zip"):
                print("extracting ", os.path.join(root, file))
                extract_zip_file(root, file)
                