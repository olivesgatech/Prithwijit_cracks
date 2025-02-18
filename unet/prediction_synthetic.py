import os
import glob
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp  # 🔹 Import SMP
from PIL import Image

class SeismicSegmentationDataset(Dataset):
    def __init__(self, image_folder, label_folder, image_transform=None, mask_transform=None, view="crossline"):
        self.image_files = sorted(glob.glob(os.path.join(image_folder, "*.dat")))
        self.label_files = sorted(glob.glob(os.path.join(label_folder, "*.dat")))
        
        assert len(self.image_files) == len(self.label_files), "Mismatch in image and label files"
        
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.view = view
        self.volumes = []
        self.labels = []
        
        for img_file, lbl_file in zip(self.image_files, self.label_files):
            img_volume = np.fromfile(img_file, dtype=np.single).reshape(128, 128, 128)
            lbl_volume = np.fromfile(lbl_file, dtype=np.single).reshape(128, 128, 128)
            xm = np.mean(img_volume)
            xs = np.std(img_volume)
            img_volume = (img_volume - xm) / xs
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
        
        # Normalize image between 0 and 1 and binarize mask
        image = (image - image.min()) / (image.max() - image.min())
        mask = (mask > 0).astype(np.float32)
        
        # Convert arrays to PIL images to use with transforms
        image = Image.fromarray((image * 255).astype(np.uint8))
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        return image, mask

def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    """Compute the Dice Coefficient."""
    y_true = y_true.float()
    y_pred = (y_pred > 0.5).float()
    
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

def save_results_locally(images, masks, preds, save_dir, num_samples=10):
    """Save sample segmentation results locally for visualization."""
    os.makedirs(save_dir, exist_ok=True)
    sample_indices = random.sample(range(images.shape[0]), min(num_samples, images.shape[0]))
    
    for i, idx in enumerate(sample_indices):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        # Display the first channel of the image (assuming seismic images)
        ax[0].imshow(images[idx, 0], cmap="seismic")
        ax[0].set_title("Input Image")
        ax[1].imshow(masks[idx].squeeze(0), cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[2].imshow(preds[idx].squeeze(0), cmap="gray")
        ax[2].set_title("Prediction")
        
        save_path = os.path.join(save_dir, f"sample_{i}.png")
        plt.savefig(save_path)
        plt.close(fig)
    
    print(f"Saved {len(sample_indices)} segmentation results in {save_dir}")

def main(args):
    # Set the CUDA device (if available)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Initialize wandb logging
    wandb.init(project="unet-cracks-testing")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the model and set it to evaluation mode
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    model.to(device)
    model.eval()
    
    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint_path}")
    
    # Define image and mask transforms
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
    
    # Create the test DataLoader
    test_dataset = SeismicSegmentationDataset(args.test_image_folder, args.test_label_folder, image_transform, mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    
    total_dice = 0
    all_images, all_masks, all_preds = [], [], []
    
    # Create directories to save prediction logits, ground truth masks, and raw predictions (npy format)
    os.makedirs(args.save_gt_folder, exist_ok=True)
    os.makedirs(args.save_pred_folder, exist_ok=True)
    
    global_idx = 0  # To keep a unique index for every sample saved
    
    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc="Evaluating Model on Test Set")
        for images, masks in test_loader_tqdm:
            images = images.to(device)
            masks = masks.to(device, dtype=torch.float32)
            
            # Forward pass
            outputs = model(images)  # Raw outputs (logits)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid for binary segmentation
            
            # Interpolate outputs to the size of the ground truth masks for evaluation
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
           
            # Compute Dice Score
            dice_score = dice_coefficient(masks, outputs)
            total_dice += dice_score.item()
            
            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            # Save thresholded predictions for later visualization
            all_preds.append((outputs > 0.5).cpu().float())
            
            # Save each ground truth mask and corresponding raw prediction (before threshold) in npy format
            batch_size = images.shape[0]
            for i in range(batch_size):
                # Get ground truth mask and raw prediction (both with shape [1, H, W])
                gt_tensor = masks[i].cpu()       # Already binarized
                pred_tensor = outputs[i].cpu()     # Raw prediction (probabilities)
                
                # Convert tensors to numpy arrays
                gt_np = gt_tensor.numpy()   # shape [1, H, W]
                pred_np = pred_tensor.numpy()   # shape [1, H, W]
                
                gt_save_path = os.path.join(args.save_gt_folder, f"mask_{global_idx:05d}.npy")
                pred_save_path = os.path.join(args.save_pred_folder, f"pred_{global_idx:05d}.npy")
                
                np.save(gt_save_path, gt_np)
                np.save(pred_save_path, pred_np)
                
                global_idx += 1

    # Compute and print the average Dice Score
    avg_dice_score = total_dice / len(test_loader)
    print(f"Average Dice Score on Test Set: {avg_dice_score:.4f}")
    
    # Concatenate all results for visualization
    all_images = torch.cat(all_images, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    
    # Save sample segmentation results locally for visualization
    save_results_locally(all_images, all_masks, all_preds, save_dir=args.save_dir)
    
    print("Evaluation complete. Results logged to wandb.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Segmentation Evaluation")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--test_image_folder", type=str, required=True,
                        help="Path to test images folder")
    parser.add_argument("--test_label_folder", type=str, required=True,
                        help="Path to test labels folder")
    parser.add_argument("--save_gt_folder", type=str, required=True,
                        help="Directory to save ground truth masks in npy format")
    parser.add_argument("--save_pred_folder", type=str, required=True,
                        help="Directory to save raw prediction maps in npy format")
    parser.add_argument("--save_dir", type=str, default="finetune_novice_pretrained_synthetic",
                        help="Directory to save segmentation result images")
    
    args = parser.parse_args()
    main(args)
