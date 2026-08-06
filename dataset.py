import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from astropy.io import fits
from utils.functions import normalize_pair


# Define dataset class to read .fits dataset
class StampsDataset(Dataset):
    def __init__(self, path_noisy, path_clean, transform=None):
        stamps_clean = list(Path(path_clean).iterdir())
        stamps_noisy = list(Path(path_noisy).iterdir())
        dir_size = len(stamps_clean)

        self.clean_imgs = []
        self.noisy_imgs = []

        for stp in range(dir_size):
            clean = fits.open(stamps_clean[stp])[0].data.astype(np.float32)
            noisy = fits.open(stamps_noisy[stp])[0].data.astype(np.float32)

            noisy, clean = normalize_pair(noisy, clean) 
            self.clean_imgs.append(clean)
            self.noisy_imgs.append(noisy)

        self.transform = transform

    def __len__(self):
        return len(self.clean_imgs)

    def __getitem__(self, idx):
        clean_tensor = torch.from_numpy(self.clean_imgs[idx]).unsqueeze(0)
        noisy_tensor = torch.from_numpy(self.noisy_imgs[idx]).unsqueeze(0)

        if self.transform:
            noisy_tensor = self.transform(noisy_tensor)
            clean_tensor = self.transform(clean_tensor)

        return noisy_tensor, clean_tensor