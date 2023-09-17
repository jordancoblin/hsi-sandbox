import os
import zipfile as zf
from pathlib import Path

ZIP_DIR = Path('./hytexila/ENVI/compressed')

def extract_zip_file(root, file):
    """Extracts a zip file to the directory"""
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
                