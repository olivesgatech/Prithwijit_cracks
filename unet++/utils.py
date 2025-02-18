import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import wandb
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp


class SeismicNumpyDataset(Dataset):
    def __init__(self, image_folder, label_folder, image_transform=None, mask_transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Get sorted list of `.npy` files
        self.image_files = sorted(glob.glob(os.path.join(image_folder, "*.npy")))
        self.label_files = sorted(glob.glob(os.path.join(label_folder, "*.npy")))

        assert len(self.image_files) == len(self.label_files), "Mismatch in images and labels!"

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        image = np.load(image_path).astype(np.float32)  # Load image
        label = np.load(label_path).astype(np.float32) # Load label

        # Normalize image between 0 and 1 and convert to PIL Image
        image = Image.fromarray(((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8))
        label = Image.fromarray((label * 255).astype(np.uint8))
        
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            label = self.mask_transform(label)
            # Ensure binary mask if needed

        # 🔹 Return the base filename so that predictions can be saved with the same name
        filename = os.path.basename(image_path)
        return image, label, filename

class SeismicNumpyDatasetThebe(Dataset):
    def __init__(self, image_folder, label_folder, image_transform=None, mask_transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Get sorted list of `.npy` files
        self.image_files = sorted(glob.glob(os.path.join(image_folder, "*.npy")))
        self.label_files = sorted(glob.glob(os.path.join(label_folder, "*.npy")))

        assert len(self.image_files) == len(self.label_files), "Mismatch in images and labels!"

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        image = np.load(image_path).astype(np.float32).T  # Load image
        label = np.load(label_path).astype(np.float32).T  # Load label

        # Normalize image between 0 and 1 and convert to PIL Image
        image = Image.fromarray(((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8))
        label = Image.fromarray((label * 255).astype(np.uint8))
        
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            label = self.mask_transform(label)
            # Ensure binary mask if needed

        # 🔹 Return the base filename so that predictions can be saved with the same name
        filename = os.path.basename(image_path)
        return image, label, filename
    
class SeismicSegmentationDataset(Dataset):
    def __init__(self, image_folder, label_folder, image_transform =None, mask_transform =None, view="crossline"):
        self.image_files = sorted(glob.glob(os.path.join(image_folder, "*.dat")))
        self.label_files = sorted(glob.glob(os.path.join(label_folder, "*.dat")))
        
        assert len(self.image_files) == len(self.label_files), "Mismatch in image and label files"
        
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.view = view
        self.volumes = []
        self.labels = []
        
        for img_file, lbl_file in zip(self.image_files, self.label_files):
            img_volume = np.fromfile(img_file, dtype=np.single).reshape(128,128,128)
            lbl_volume = np.fromfile(lbl_file, dtype=np.single).reshape(128,128,128)
            xm = np.mean(img_volume)
            xs = np.std(img_volume)
            img_volume = img_volume - xm
            img_volume = img_volume / xs
            # img_volume = np.transpose(img_volume)
            # lbl_volume = np.transpose(lbl_volume)
            self.volumes.append(img_volume)
            self.labels.append(lbl_volume)

        # Compute total number of slices
        self.slices_per_volume = self.volumes[0].shape[0] if self.view == "crossline" else self.volumes[0].shape[1]
        self.total_slices = len(self.volumes) * self.slices_per_volume
    
    def __len__(self):
        return self.total_slices
    
    def __getitem__(self, idx):
        # Identify the volume and slice index
        volume_idx = idx // self.slices_per_volume
        slice_idx = idx % self.slices_per_volume
        
        img_volume = self.volumes[volume_idx]
        lbl_volume = self.labels[volume_idx]
        
        if self.view == "crossline":
            image = img_volume[slice_idx, :, :]
            mask = lbl_volume[slice_idx, :, :]
        else:  # Inline view
            image = img_volume[:, slice_idx, :]
            mask = lbl_volume[:, slice_idx, :]
        
        # Normalize image and binarize mask
        image = (image - image.min()) / (image.max() - image.min())  # Normalize between 0 and 1
        mask = (mask > 0).astype(np.float32)  # Convert to binary mask
        
        image = Image.fromarray((image * 255).astype(np.uint8))
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        
        if self.image_transform  :
            image = self.image_transform (image)
        
        if self.mask_transform:
            mask = self.mask_transform (mask)
            mask = (mask > 0.5).float()
        return image, mask