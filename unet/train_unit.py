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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Initialize Weights & Biases
wandb.init(project="unet-resnet50-training")


def dice_loss_with_empty_handling(y_true, y_pred, epsilon=1e-6):
    """
    Dice loss with handling for samples with no positive pixels.

    Args:
        y_true: Tensor of ground truth labels (binary: 0 or 1).
        y_pred: Tensor of predicted probabilities.
        epsilon: Small constant for numerical stability.

    Returns:
        Dice loss value.
    """
    y_true = y_true.float()  # Ensure float32
    y_pred = torch.sigmoid(y_pred)  # Convert logits to probabilities

    # Flatten tensors
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)

    # Compute intersection and union
    intersection = torch.sum(y_true_flat * y_pred_flat)
    denominator = torch.sum(y_true_flat) + torch.sum(y_pred_flat)

    # Handle cases where both y_true and y_pred are empty
    if denominator == 0:
        dice_coeff = torch.tensor(1.0, device=y_true.device)  # No positives, perfect match
    else:
        dice_coeff = (2. * intersection + epsilon) / (denominator + epsilon)

    dice_loss_value = 1.0 - dice_coeff
    return dice_loss_value

# Function to Log Segmentation Results to wandb
def log_segmentation_results(images, masks, preds, num_samples=4):
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()
    
    log_images = []
    
    for i in range(min(num_samples, images.shape[0])):        
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        
        ax[0].imshow(images[i, 0], cmap="gray")  # Original Seismic Image
        ax[0].set_title("Seismic Image")
        
        ax[1].imshow(masks[i], cmap="gray")  # Ground Truth
        ax[1].set_title("Ground Truth")
        
        ax[2].imshow(preds[i], cmap="gray")  # Model Prediction
        ax[2].set_title("Model Prediction")
        
        plt.tight_layout()
        
        # Log figure as wandb image
        log_images.append(wandb.Image(fig))
        plt.close(fig)
    
    wandb.log({"Segmentation Samples": log_images})


# ---------------------------
# 🔹 Dataset Class
# ---------------------------
# Custom Dataset for Seismic Binary Segmentation (Loads Multiple Volumes)
# Custom Dataset for Seismic Binary Segmentation (Loads Multiple Volumes)
class SeismicSegmentationDataset(Dataset):
    def __init__(self, image_folder, label_folder, image_transform =None, mask_transform =None, view="crossline"):
        self.image_files = sorted(glob.glob(os.path.join(image_folder, "*.dat")))
        self.label_files = sorted(glob.glob(os.path.join(label_folder, "*.dat")))
        
        assert len(self.image_files) == len(self.label_files), "Mismatch in image and label files"
        
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.view = view  # "crossline" or "inline"
        self.volumes = []
        self.labels = []
        
        # Load all 3D volumes into memory
        for img_file, lbl_file in zip(self.image_files, self.label_files):
            img_volume = np.fromfile(img_file, dtype=np.single).reshape(128,128,128)
            lbl_volume = np.fromfile(lbl_file, dtype=np.single).reshape(128,128,128)
            xm = np.mean(img_volume)
            xs = np.std(img_volume)
            # Normalize image volume
            img_volume = img_volume - xm
            img_volume = img_volume / xs
            # Transpose image and label volumes
            img_volume = np.transpose(img_volume)
            lbl_volume = np.transpose(lbl_volume)
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

# ---------------------------
# 🔹 Transformations
# ---------------------------
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:1, :, :])
])

# Load Dataset (Extracting both crossline & inline views from all volumes)
image_folder = "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/train/seis"
label_folder = "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/train/fault"

crossline_dataset = SeismicSegmentationDataset(image_folder, label_folder, image_transform, mask_transform, view="crossline")
inline_dataset = SeismicSegmentationDataset(image_folder, label_folder, image_transform, mask_transform, view="inline")

full_dataset = ConcatDataset([crossline_dataset, inline_dataset])

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1, pin_memory=True)

print(f"Loaded {len(full_dataset)} total slices from {len(crossline_dataset.image_files)} volumes.")

# ---------------------------
# 🔹 Initialize Model (U-Net with ResNet-50 Backbone)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load U-Net with ResNet-50 Backbone
model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
model.to(device)

# criterion = nn.BCEWithLogitsLoss()
def criterion(outputs, masks):
    return dice_loss_with_empty_handling(masks, outputs)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scaler = GradScaler()

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

# ---------------------------
# 🔹 Checkpoint Management
# ---------------------------
best_val_loss = float("inf")
patience = 5
early_stop_count = 0
checkpoint_path = "./checkpoints/unet_resnet50_checkpoint.pth"
best_model_path = "./checkpoints/best_unet_resnet50.pth"

# Resume training if checkpoint exists
if os.path.exists(checkpoint_path):
    print("Resuming training from checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    best_val_loss = checkpoint["best_val_loss"]

# ---------------------------
# 🔹 Training Loop with Early Stopping & Checkpoints
# ---------------------------
# ---------------------------
# 🔹 Training Loop with tqdm
# ---------------------------
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_norm = 0
    
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
    for images, masks in train_loader_tqdm:
        images, masks = images.to(device), masks.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()

        total_norm += torch.norm(torch.stack([
            torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None
        ]), 2).item()

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
    
    avg_train_loss = total_loss / len(train_loader)
    avg_gradient_norm = total_norm / len(train_loader)

    # Validation Phase
    model.eval()
    val_loss = 0
    all_images, all_masks, all_preds = [], [], []
    
    val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        # for images, masks in val_loader_tqdm:
        for batch_idx, (images, masks) in enumerate(val_loader_tqdm):
            images, masks = images.to(device), masks.to(device, dtype=torch.float32)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_loader_tqdm.set_postfix(loss=loss.item())
            # all_images.append(images.cpu())
            # all_masks.append(masks.cpu())
            # all_preds.append((outputs > 0.5).cpu().float())
            

    avg_val_loss = val_loss / len(val_loader)


    
    
    
    scheduler.step(avg_val_loss)
    wandb.log({
        "Train Loss": avg_train_loss,
        "Val Loss": avg_val_loss,
        "Gradient Norm": avg_gradient_norm,
        "Learning Rate": optimizer.param_groups[0]["lr"]})
    
    # all_images = torch.cat(all_images, dim=0)
    # all_masks = torch.cat(all_masks, dim=0)
    # all_preds = torch.cat(all_preds, dim=0)
    
    # log_segmentation_results(all_images, all_masks, all_preds)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Gradient Norm: {avg_gradient_norm:.4f}")


    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }, checkpoint_path)
    
    # if avg_val_loss >= best_val_loss:
    #     early_stop_count += 1
    #     if early_stop_count >= patience:
    #         print("Early stopping triggered.")
    #         break

    if epoch >= 5:  # Ensure early stopping is applied only after 5 epochs
        if avg_val_loss >= best_val_loss:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping triggered.")
                break
