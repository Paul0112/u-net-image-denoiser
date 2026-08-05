# Functions to add noise in ZTF stamps
from astropy.io import fits
from pathlib import Path
import numpy as np

def add_gaussian_noise(img_path, OUTPUT_DIR= None, mean= 0, std= 25):
    # Open a .fits file and add gaussian noise to it
    oid = Path(img_path).stem
    save_path = f"{OUTPUT_DIR}/{oid}_noisy.fits"
    if Path(save_path).exists(): return save_path
    
    hdu = fits.open(img_path)
    img = hdu[0].data
    header = hdu[0].header

    noise = np.random.normal(mean, std, img.shape)
    noisy_image = img + noise
    
    if OUTPUT_DIR != None: fits.writeto(save_path, noisy_image, header, overwrite=True)
    hdu.close()

    return noisy_image