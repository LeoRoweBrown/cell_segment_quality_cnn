import numpy as np
from glob import glob
import os
import torch
from torch.utils.data import Dataset
import tifffile as tiff
from warnings import warn
import json

class CombinedTrainingSet(Dataset):
    def __init__(self, image_dataset, label_dataset):
        self.image_dataset = image_dataset
        self.label_dataset = label_dataset
        print(len(image_dataset), len(label_dataset))
        assert len(image_dataset) == len(label_dataset), "Datasets must be same length!"

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image = self.image_dataset[idx]
        label = self.label_dataset[idx]
        return image, label

class ClassificationDataset(Dataset):
    def __init__(self, json_file_path, transform=None):
        self.classification_data = []
        self.append_json_dataset(json_file_path)
        self.transform = transform
        print("JSON data set has length", len(self.classification_data))
    
    def __len__(self):
        return len(self.classification_data)

    def __getitem__(self, idx):
        """Get the rating from json/dict and cast to float for compatibility with model"""
        return torch.tensor(self.classification_data[idx]['rating'], dtype=torch.float32)  # Ensures float32

    def append_json_dataset(self, json_file_path):
        with open(json_file_path, 'r') as file:
            dataset = json.load(file)
        self.classification_data += dataset

class Tiff3DDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        
        # List all data files (assuming consistent numbering)
        self.data_files = sorted(
            glob(os.path.join(data_dir, 'data_roi*.tif'))
        )
        self.mask_files = sorted(
            glob(os.path.join(data_dir, 'mask_roi*.tif'))
        )
        self.data_files = []
        for mask_f in self.mask_files:
            data_f = mask_f.replace('mask_roi', 'data_roi')
            if os.path.exists(data_f):
                self.data_files.append(data_f)

        print("Found ", len(self.data_files), "rois")

        if len(self.data_files) != len(self.mask_files):
            warn(f"{len(self.data_files)} fluorescence stacks but {len(self.mask_files)} mask stacks")
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        mask_path = self.mask_files[idx]
        
        # Load tiff volumes (will be numpy arrays)
        data = tiff.imread(data_path)  # Shape is z, y, x
        mask = tiff.imread(mask_path)  # 
        
        # Convert to torch tensors and normalize. TODO: should we normalize?
        data = torch.from_numpy(data).float()
        mask = torch.from_numpy(mask).float()
        
        # Add channel dimension
        data = data.unsqueeze(0)  # [1, z, y, x]  (1 channel)
        mask = mask.unsqueeze(0)  # [1, z, y, x]  (1 channel)
        
        if self.transform:
            data, mask = self.transform(data, mask)

        # Concat together
        data_and_mask = torch.cat([data, mask], dim=0)
        
        return data_and_mask

    def append_dataset(self, dataset):
        new_data_files = dataset.data_files
        new_mask_files = dataset.mask_files
        if hasattr(new_mask_files, "len"):
            self.data_files += new_data_files
            self.mask_files += new_mask_files
        else:
            self.data_files.append(new_data_files)  # for length 1 data, very rare
            self.mask_files.append(new_mask_files)