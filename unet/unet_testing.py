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
import segmentation_models_pytorch as smp  # 🔹 Import SMP

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Initialize Weights & Biases
wandb.init(project="unet-resnet50-testing")

# ---------------------------
# 🔹 Load SMP U-Net Model (ResNet50 Backbone WITHOUT IMAGENET WEIGHTS)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "/home/prithwijit/Cracks/unet/checkpoints/best_unet_resnet50.pth"

# Initialize the U-Net model (No ImageNet Weights)
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,  # ❌ No ImageNet weights
    in_channels=3,
    classes=1,  # Binary segmentation (1 output channel)
    activation=None  # We apply sigmoid manually
).to(device)

# Load your trained checkpoint
if os.path.exists(checkpoint_path):
    print(f"Loading best model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
else:
    raise FileNotFoundError("Best model checkpoint not found!")

# ---------------------------
# 🔹 Dataset Definition (No Changes)
# ---------------------------
class SeismicSegmentationDataset(Dataset):
    def __init__(self, image_folder, label_folder, image_transform=None, mask_transform=None, view="crossline"):
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
            img_volume = np.fromfile(img_file, dtype=np.single).reshape(128, 128, 128)
            lbl_volume = np.fromfile(lbl_file, dtype=np.single).reshape(128, 128, 128)
            
            # Normalize image volume
            xm = np.mean(img_volume)
            xs = np.std(img_volume)
            img_volume = (img_volume - xm) / xs
            
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
        
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        
        return image, mask

# ---------------------------
# 🔹 Define Transformations (No Changes)
# ---------------------------
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Repeat to make it 3-channel
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:1, :, :])  # Keep only 1 channel
])

# ---------------------------
# 🔹 Create Test DataLoader (No Changes)
# ---------------------------
test_image_folder = "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/validation/seis"
test_label_folder = "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/validation/fault"

test_dataset = SeismicSegmentationDataset(test_image_folder, test_label_folder, image_transform=image_transform, mask_transform=mask_transform, view="crossline")
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

# ---------------------------
# 🔹 Dice Score Computation (No Changes)
# ---------------------------
def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    y_true = y_true.float()
    y_pred = (y_pred > 0.5).float()
    
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

# ---------------------------
# 🔹 Evaluate Model
# ---------------------------
total_dice = 0
all_images, all_masks, all_preds = [], [], []

with torch.no_grad():
    test_loader_tqdm = tqdm(test_loader, desc="Evaluating Model on Test Set")
    
    for images, masks in test_loader_tqdm:
        images, masks = images.to(device), masks.to(device, dtype=torch.float32)
        
        outputs = model(images)  # 🔹 U-Net forward pass
        outputs = torch.sigmoid(outputs)  # Apply sigmoid for binary segmentation
        outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
        
        dice_score = dice_coefficient(masks, outputs)
        total_dice += dice_score.item()
        
        all_images.append(images.cpu())
        all_masks.append(masks.cpu())
        all_preds.append((outputs > 0.5).cpu().float())

# Compute Average Dice Score
avg_dice_score = total_dice / len(test_loader)
print(f"Average Dice Score on Test Set: {avg_dice_score:.4f}")

# ---------------------------
# 🔹 Save Segmentation Results (No Changes)
# ---------------------------
all_images = torch.cat(all_images, dim=0)
all_masks = torch.cat(all_masks, dim=0)
all_preds = torch.cat(all_preds, dim=0)

def save_results_locally(images, masks, preds, save_dir="segmentation_results", num_samples=10):
    os.makedirs(save_dir, exist_ok=True)
    
    sample_indices = random.sample(range(images.shape[0]), min(num_samples, images.shape[0]))
    
    for i, idx in enumerate(sample_indices):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(images[idx, 0], cmap="seismic")
        ax[1].imshow(masks[idx].squeeze(0), cmap="gray")
        ax[2].imshow(preds[idx].squeeze(0), cmap="gray")
        
        save_path = os.path.join(save_dir, f"sample_{i}.png")
        plt.savefig(save_path)
        plt.close(fig)
    
    print(f"Saved {len(sample_indices)} segmentation results in {save_dir}")

save_results_locally(all_images, all_masks, all_preds)
print("Evaluation complete. Results logged to wandb.")
