# Functions/utils used in the notebooks
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path


def download_stamp(oid, client, output_dir, to_delete):
    """
    Download a stamp from the ALeRCE cliente based on the 
    oid provided. If the oid provided present a exception, 
    it's saved in a 'to_delete' list to delete it from the dataset

    """
    file_path = os.path.join(output_dir, f"{oid}_stamps.fits")

    if os.path.exists(file_path):
        return True 

    try:
        stamps = client.get_stamps(oid)
        stamps.writeto(file_path, overwrite=True)
        time.sleep(0.05)
        return True

    except:
        to_delete.append(oid) 
        return False


def visual_fits(path):
    '''
    """
    Helper function for visualizing the three
    images stored in a .fits stamp file.
    """
    '''
    hdu = fits.open(path)

    science = hdu[0].data
    reference = hdu[1].data
    difference = hdu[2].data

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  

    # sCIENCE
    axes[0].imshow(science, cmap="gray")
    axes[0].set_title("Science")
    axes[0].axis('off') 

    # REFERENCE
    axes[1].imshow(reference, cmap="gray")
    axes[1].set_title("Reference")
    axes[1].axis('off') 

    # DIFFERENCE
    axes[2].imshow(difference, cmap="gray")
    axes[2].set_title("Difference")
    axes[2].axis('off') 

    plt.show()
    hdu.close()


def clean_stamps(to_delete, output_dir):
    '''
    """
    Helper function that continues the corrupted
    OID filtering process started during the
    download. It appends to `to_delete` any
    entries whose stamps do not have the expected
    shape (63, 63, 3).
    '''
    to_delete = []
    clean_list = list(Path(output_dir).iterdir())
    len_clean = len(clean_list)

    for stp in range(len_clean):
        stamp_path = clean_list[stp]
        hdu = fits.open(stamp_path)
        stamp = np.stack([hdu[0].data, hdu[1].data, hdu[2].data], axis=-1)

        if stamp.shape != (63,63,3):
            to_delete.append(stamp_path)
            hdu.close()

    for stamp_path in to_delete:
        os.remove(stamp_path)


def normalize_pair(noisy, clean):
    """
    Apply L_2 normalization of clean stamps
    to both noisy and clean images
    """
    clean = np.nan_to_num(clean.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    noisy = np.nan_to_num(noisy.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    norm = np.linalg.norm(clean)
    norm = norm if norm != 0 else 1

    return noisy / norm, clean / norm



def plot_training_history(train_loss_history, val_loss_history):
    """
    Helper function for ploting the loss between validation and training set 
    """
    plt.figure(figsize=(12, 5))

    # LOSS
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Learning curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()