import numpy as np
from glob import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff
from warnings import warn

class Tiff3DDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # List all data files (assuming consistent numbering)
        self.data_files = sorted(
            glob.glob(os.path.join(data_dir, 'data_roi*.tif'))
        )
        self.mask_files = sorted(
            glob.glob(os.path.join(data_dir, 'mask_roi*.tif'))
        )
        if len(self.data_files) != len(self.mask_files):
            warn(f"{len(self.data_files)} fluorescence stacks but {len(self.mask_files)} mask stacks")
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        mask_path = os.path.join(self.data_dir, self.mask_files[idx])
        
        # Load TIFF volumes (will be numpy arrays)
        data = tiff.imread(data_path)  # Shape is z, y, x (or D, H, W)
        mask = tiff.imread(mask_path)  # 
        
        # Convert to torch tensors and normalize if needed
        data = torch.from_numpy(data).float()
        mask = torch.from_numpy(mask).float()
        
        # Optional: Add channel dimension if needed
        data = data.unsqueeze(0)  # [1, z, y, x]  (1 channel)
        mask = mask.unsqueeze(0)  # [1, z, y, x]  (1 channel)

        if self.transform:
            data, mask = self.transform(data, mask)
        
        return data, mask
