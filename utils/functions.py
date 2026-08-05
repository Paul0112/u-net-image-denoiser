# Functions/utils used in the notebooks
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


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


def clean_labels(labels, to_delete, output_dir):
    '''
    """
    Helper function that continues the corrupted
    OID filtering process started during the
    download. It appends to `to_delete` any
    entries whose stamps do not have the expected
    shape (63, 63, 3) and returns the labels.
    '''
    for idx, row in labels.iterrows():
        oid = row['oid']
        label = row['label']

        if oid not in to_delete:
            hdu = fits.open(os.path.join(output_dir, f"{oid}_stamps.fits"))
            stamp = np.stack([hdu[0].data, hdu[1].data, hdu[2].data], axis=-1)

            if stamp.shape != (63,63,3):
                to_delete.append(oid)

    labels_cleaned = labels[~labels['oid'].isin(to_delete)].reset_index(drop=True)
    return labels_cleaned
