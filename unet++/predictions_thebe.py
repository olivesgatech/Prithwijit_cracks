import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import SeismicNumpyDataset, SeismicSegmentationDataset, SeismicNumpyDatasetThebe
import segmentation_models_pytorch as smp  # 🔹 Import SMP

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Initialize wandb logging
    wandb.init(project="unetPP-cracks-testing")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the model
    model = smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    model.to(device)
    
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
    test_dataset = SeismicNumpyDatasetThebe(args.test_image_folder, args.test_label_folder, image_transform, mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    
    total_dice = 0
    all_images, all_masks, all_preds = [], [], []
    
    # Create the directory to save prediction logits
    os.makedirs(args.prediction_logits_dir, exist_ok=True)
    
    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc="Evaluating Model on Test Set")
        for images, masks, filenames in test_loader_tqdm:
            images, masks = images.to(device), masks.to(device, dtype=torch.float32)
            
            # Forward pass
            outputs = model(images)  # Raw outputs (logits)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid for binary segmentation
            
            # Interpolate outputs to a specific size for saving logits
            save_outputs = F.interpolate(outputs, size=(512, 512), mode="bilinear", align_corners=False)
            # Interpolate outputs to the size of the ground truth masks for evaluation
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
           
            # Compute Dice Score
            dice_score = dice_coefficient(masks, outputs)
            total_dice += dice_score.item()
            
            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            all_preds.append((outputs > 0.5).cpu().float())
            
            # Save each sample's prediction logits using its filename
            for i, fname in enumerate(filenames):
                pred_logits = save_outputs[i].cpu().numpy().squeeze(0)  # Shape: (H, W)
                save_path = os.path.join(args.prediction_logits_dir, fname)
                np.save(save_path, pred_logits)
    
    # Compute and print the average Dice Score
    avg_dice_score = total_dice / len(test_loader)
    print(f"Average Dice Score on Test Set: {avg_dice_score:.4f}")
    
    # Concatenate all results for visualization
    all_images = torch.cat(all_images, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    
    # Save sample segmentation results locally
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
    parser.add_argument("--prediction_logits_dir", type=str, default="prediction_logits_finetune_novice_pretrained_synthetic",
                        help="Directory to save prediction logits")
    parser.add_argument("--save_dir", type=str, default="finetune_novice_pretrained_synthetic",
                        help="Directory to save segmentation result images")
    
    args = parser.parse_args()
    main(args)
