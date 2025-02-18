import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import segmentation_models_pytorch as smp
from PIL import Image
import wandb
from torch.cuda.amp import autocast, GradScaler
import glob
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# ----------------------
parser = argparse.ArgumentParser(description="Fine-tune Segformer for seismic fault segmentation")
parser.add_argument("--project", type=str, default="segformer-finetuning-thebe-dice-flat", help="WandB project name")
parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing seismic images in .npy format")
parser.add_argument("--label_folder", type=str, required=True, help="Path to folder containing seismic labels in .npy format")
parser.add_argument("--resume_checkpoint", type=str, default="", help="Path to a checkpoint to resume training from")
parser.add_argument("--best_checkpoint", type=str, default="best_segformer_thebe_finetune.pth", help="File path to save the best model")
parser.add_argument("--current_checkpoint", type=str, default="segformer_thebe_finetune.pth", help="File path to save the current checkpoint")
args = parser.parse_args()

##################################################


# Restrict PyTorch to use only GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Weights & Biases
wandb.init(project=args.project)

def compute_mean_std(image_folder):
    """ Compute global mean and std of all images in the folder """
    all_files = sorted(glob.glob(os.path.join(image_folder, "*.npy")))
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    num_pixels = 0

    print("Computing global mean and std for normalization...")
    for file in tqdm(all_files):
        img = np.load(file).astype(np.float32)
        pixel_sum += img.sum()
        pixel_sq_sum += (img ** 2).sum()
        num_pixels += img.size

    mean = pixel_sum / num_pixels
    std = np.sqrt((pixel_sq_sum / num_pixels) - (mean ** 2))

    print(f"Computed Mean: {mean:.5f}, Std: {std:.5f}")
    return mean, std

def dice_loss_with_empty_handling(y_true, y_pred, epsilon=1e-6):
    """
    Dice loss with handling for samples with no positive pixels.    Args:
        y_true: Tensor of ground truth labels (binary: 0 or 1).
        y_pred: Tensor of predicted probabilities (requires_grad=True).
        epsilon: Small constant for numerical stability.    Returns:
        Dice loss value.
    """
    y_true = y_true.float()
    y_pred = y_pred.float()    # Flatten tensors
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)    # Compute intersection and denominator
    intersection = torch.sum(y_true_flat * y_pred_flat)
    denominator = torch.sum(y_true_flat) + torch.sum(y_pred_flat)    # Ensure denominator does not cause division by zero
    dice_coeff = (2. * intersection + epsilon) / (denominator + epsilon)    # Ensure loss remains part of computational graph
    dice_loss_value = 1.0 - dice_coeff    
    return dice_loss_value  # This remains differentiable

def generalized_dice_loss_fp_penalized(y_true, y_pred, alpha=2.0, epsilon=1e-6):
    """
    Generalized Dice Loss with False Positive penalty.

    Args:
        y_true: Tensor of ground truth labels (binary: 0 or 1).
        y_pred: Tensor of predicted probabilities.
        alpha: Scaling factor for penalizing false positives.
        epsilon: Small constant for numerical stability.

    Returns:
        Dice loss value with weighted FP penalty.
    """
    y_true = y_true.float()
    y_pred = torch.sigmoid(y_pred)  # Convert logits to probabilities

    # Compute per-class weights (inverse frequency)
    foreground_pixels = torch.sum(y_true, dim=(1, 2, 3)) + epsilon  # Avoid div-by-zero
    background_pixels = torch.sum(1 - y_true, dim=(1, 2, 3)) + epsilon
    w_foreground = 1.0 / (foreground_pixels ** 2)  # Higher weight for sparse foreground
    w_background = 1.0 / (background_pixels ** 2)  # Lower weight for background

    # Compute intersection and denominator
    intersection = torch.sum(w_foreground * torch.sum(y_true * y_pred, dim=(1, 2, 3)))
    
    # Denominator includes false positives (weighted by alpha)
    false_positives = torch.sum(w_background * torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3)))
    denominator = torch.sum(w_foreground * torch.sum(y_true + y_pred, dim=(1, 2, 3))) + alpha * false_positives

    dice_coeff = (2. * intersection + epsilon) / (denominator + epsilon)
    
    return 1.0 - dice_coeff  # Loss is 1 - Dice

def log_segmentation_results(images, masks, preds, num_samples=4):
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()
    
    log_images = []
    
    for i in range(min(num_samples, images.shape[0])):        
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        
        ax[0].imshow(images[i, 0], cmap="seismic")  # Original Seismic Image
        ax[0].set_title("Seismic Image")
        
        # ax[1].imshow(masks[i], cmap="gray")  # Ground Truth
        ax[1].imshow(masks[i].squeeze(0), cmap="gray")
        ax[1].set_title("Ground Truth")
        
        # ax[2].imshow(preds[i], cmap="gray")  # Model Prediction
        ax[2].imshow(preds[i].squeeze(0), cmap="gray")
        ax[2].set_title("Model Prediction")
        
        plt.tight_layout()
        
        # Log figure as wandb image
        log_images.append(wandb.Image(fig))
        plt.close(fig)
    
    wandb.log({"Segmentation Samples": log_images})

class SeismicNumpyDataset(Dataset):
    def __init__(self, image_folder, label_folder, mean, std, image_transform =None, mask_transform =None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.mean = mean
        self.std = std
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

        # Z-score Normalization
        # image = (image - self.mean) / self.std
        # label = (label > 0.5).astype(np.float32)  # Binarize labels

        # Convert numpy arrays to PIL images
        image = Image.fromarray(((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8))
        label = Image.fromarray((label * 255).astype(np.uint8))
 
        
        if self.image_transform  :
            image = self.image_transform (image)
        
        if self.mask_transform:
            label = self.mask_transform (label)
             # Ensure binary mask

        return image, label


# Define Transforms
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:1, :, :])
])

# Load Dataset
image_folder = args.image_folder
label_folder = args.label_folder

global_mean, global_std = compute_mean_std(image_folder)

dataset = SeismicNumpyDataset(image_folder, label_folder, global_mean, global_std, image_transform, mask_transform)

# Train-validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1, pin_memory=True)

print(f"Loaded {len(dataset)} total samples from {image_folder}.")

# Load Model
# Load U-Net with ResNet-50 Backbone
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load U-Net with ResNet-50 Backbone
model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
model.to(device)


def criterion(outputs, masks):
    return dice_loss_with_empty_handling(masks, outputs)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)


checkpoint_path = args.resume_checkpoint


# Resume training if checkpoint exists
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# Fine-Tuning
# Early Stopping Parameters
patience = 5  # Number of epochs to wait before stopping
early_stop_count = 0
best_val_loss = float("inf")

# Fine-Tuning
print("Starting fine-tuning...")
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        images, masks = images.to(device), masks.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    all_images, all_masks, all_preds = [], [], []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device, dtype=torch.float32)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            all_preds.append((outputs > 0.5).cpu().float())  # Binarize predictions

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    wandb.log({
        "Train Loss": avg_train_loss,
        "Validation Loss": avg_val_loss,
        "Learning Rate": optimizer.param_groups[0]["lr"]
    })

    all_images = torch.cat(all_images, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    
    log_segmentation_results(all_images, all_masks, all_preds)

    scheduler.step(avg_val_loss)
    torch.save(model.state_dict(), args.current_checkpoint)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), args.best_checkpoint)
        print("New best model saved!")
        early_stop_count = 0  # Reset early stopping counter
    else:
        early_stop_count += 1

    # Check if early stopping should be triggered
    if early_stop_count >= patience:
        print("Early stopping triggered. Stopping training.")
        break
