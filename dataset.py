import os
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from astropy import fits

# Define dataset class to read .fits dataset
class StampsDataset(Dataset):
    def __init__(self, path_noisy, path_clean, transform=None):
        self.img_clean = list(Path(path_clean).iterdir())
        self.img_noisy = list(Path(path_noisy).iterdir())
        self.transform = transform

    def __len__(self):
        return len(self.img_clean)

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        noisy_path = self.noisy_files[idx]

        # Open .fits an extract only science image
        hdu_clean = fits.open(clean_path)
        hdu_noisy = fits.open(noisy_path)
        science_clean = hdu_clean[0].data
        science_noisy = hdu_noisy[0].data

        # reshpe (63, 63) to (1, 63, 63)
        clean_img = np.expand_dims(science_clean, axis=0)
        noisy_img = np.expand_dims(science_noisy, axis=0)

        clean_tensor = torch.from_numpy(clean_img)
        noisy_tensor = torch.from_numpy(noisy_img)

        if self.transform:
            noisy_tensor = self.transform(noisy_tensor)
            clean_tensor = self.transform(clean_tensor)

        hdu_clean.close()
        hdu_noisy.close()
        return noisy_tensor, clean_tensor