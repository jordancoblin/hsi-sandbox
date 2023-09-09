import spectral as sp
import spectral.io.envi as envi
import zipfile as zf
from pathlib import Path
import matplotlib.pyplot as plt

# PATH of hyperspectral images
DATA_DIR = Path('./hytexila/ENVI/wood')
SAMPLE = 'wood_01'

# Hyperspectral image filename (without extension)

if __name__ == "__main__":

    with zf.ZipFile((DATA_DIR / SAMPLE).with_suffix(".zip")) as archive:
        # print(archive.namelist())
        img = envi.open((DATA_DIR / SAMPLE).with_suffix(".hdr"), (DATA_DIR / SAMPLE).with_suffix(".raw"))
        # imgfile = archive.open('wood_01.hdr')
        # archive.extract('wood_01.hdr', DATA_DIR)
        # archive.extract('wood_01.raw', DATA_DIR)
    
    print(img)
    if img.metadata.get('default bands'):
        bands = img.metadata.get('default bands')
        iR = int(bands[0])
        iG = int(bands[1])
        iB = int(bands[2])
        view = sp.imshow(img, (iR, iG, iB), interpolation='none', title=SAMPLE)
        # plt.show()

        # view = sp.imshow(img)
        # print(view)

        # rgb = sp.get_rgb(img, (iR, iG, iB))
        # plt.imshow(rgb)
        # plt.show()

