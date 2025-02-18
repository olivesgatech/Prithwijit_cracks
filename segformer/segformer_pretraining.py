import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from PIL import Image
import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
import glob
import os
import torch.nn.functional as F
from tqdm import tqdm


# Restrict PyTorch to use only GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Weights & Biases
wandb.init(project="segformer-training")

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

# Define Transformations
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Ensure masks are the same size
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:1, :, :])   # Keep only 1 channel
])


# Load Dataset (Extracting both crossline & inline views from all volumes)
image_folder = "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/train/seis"
label_folder = "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/train/fault"

crossline_dataset = SeismicSegmentationDataset(image_folder, label_folder, image_transform =image_transform , mask_transform = mask_transform, view="crossline")
inline_dataset = SeismicSegmentationDataset(image_folder, label_folder, image_transform =image_transform , mask_transform = mask_transform, view="inline")

# Combine both views into one dataset
full_dataset = ConcatDataset([crossline_dataset, inline_dataset])

# Split datasets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

print(f"Loaded {len(full_dataset)} total slices from {len(crossline_dataset.image_files)} volumes.")

# The rest of the training pipeline remains the same...


# Define Model from Scratch
config = SegformerConfig(num_labels=1)
print("Model is configured")
model = SegformerForSemanticSegmentation(config)
print("Model is built")
# Modify Segmentation Head for Binary Segmentation
model.decode_head.classifier = nn.Conv2d(in_channels=model.config.hidden_sizes[-1], out_channels=1, kernel_size=(1, 1))
model.config.num_labels = 1
print("Model is ready for binary segmentation")
# Define Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
print("Model is ready for optimization")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model is sent to GPU")
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

# Checkpoint Saving
best_val_loss = float("inf")
patience, early_stop_count = 5, 0
resume_checkpoint = "./checkpoints/checkpoint.pth"
if os.path.exists(resume_checkpoint):
    print("Resuming training from checkpoint...")
    model.load_state_dict(torch.load(resume_checkpoint))

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


print("Training starts")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_norm = 0
    
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
    for images, masks in train_loader_tqdm:
        images, masks = images.to(device), masks.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits  # Get model outputs
        outputs = torch.sigmoid(outputs)  # Sigmoid for binary classification
        outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
        loss = criterion(outputs, masks)
        loss.backward()

        # Compute gradient norm
        total_norm += torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2).item()
        optimizer.step()
        total_loss += loss.item()
        
        train_loader_tqdm.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
    
    avg_train_loss = total_loss / len(train_loader)
    avg_gradient_norm = total_norm / len(train_loader)

    # Validation Loop
    model.eval()
    val_loss = 0
    all_images, all_masks, all_preds = [], [], []
    
    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
        
        for images, masks in val_loader_tqdm:
            images, masks = images.to(device), masks.to(device, dtype=torch.float32)
            outputs = model(pixel_values=images).logits
            preds = torch.sigmoid(outputs)  # Convert logits to probability
            preds = F.interpolate(preds, size=masks.shape[2:], mode="bilinear", align_corners=False)
            loss = criterion(preds, masks)
            val_loss += loss.item()
            
            val_loader_tqdm.set_postfix(loss=loss.item())
            
            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            all_preds.append((preds > 0.5).cpu().float())  # Binarize predictions

    avg_val_loss = val_loss / len(val_loader)

    # Log additional metrics to wandb
    wandb.log({
        "Train Loss": avg_train_loss,
        "Validation Loss": avg_val_loss,
        "Learning Rate": optimizer.param_groups[0]["lr"]
    })

    # Log segmentation visualization
    all_images = torch.cat(all_images, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    
    log_segmentation_results(all_images, all_masks, all_preds)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Gradient Norm: {avg_gradient_norm:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    scheduler.step(avg_val_loss)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "./checkpoints/best_model.pth")

    # Save last checkpoint
    torch.save(model.state_dict(), "./checkpoints/checkpoint.pth")

    # Early stopping
    if avg_val_loss >= best_val_loss:
        early_stop_count += 1
        if early_stop_count >= patience:
            print("Early stopping triggered.")
            break
    else:
        early_stop_count = 0
