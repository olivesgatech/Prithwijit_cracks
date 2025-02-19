import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def stitch_patches_to_volume(patch_dir, num_slices, original_height, original_width, patch_size=(254,254), stride=50):
    patch_height, patch_width = patch_size
    # Compute the padding that was added during extraction.
    pad_amt = (patch_width - (original_width % patch_width)) % patch_width
    padded_width = original_width + pad_amt
    
    # Initialize accumulators and a count array for averaging overlapping regions.
    volume_accum = np.zeros((num_slices, original_height, padded_width))
    volume_count = np.zeros((num_slices, original_height, padded_width))
    
    # List all patch files (assumes naming scheme: "sliceIndex_x.npy")
    patch_files = sorted(glob.glob(os.path.join(patch_dir, "*.npy")))
    
    # Wrap the patch file iteration with tqdm for a progress bar.
    for patch_file in tqdm(patch_files, desc="Stitching patches"):
        basename = os.path.basename(patch_file)
        # Filename format is assumed to be "sliceIdx_x.npy"
        name_part = os.path.splitext(basename)[0]
        slice_idx_str, x_str = name_part.split("_")
        slice_idx = int(slice_idx_str)
        x_coord = int(x_str)
        
        patch = np.load(patch_file)
        
        # Add the patch's contribution to the correct region in the accumulator.
        volume_accum[slice_idx, :, x_coord:x_coord+patch_width] += patch
        volume_count[slice_idx, :, x_coord:x_coord+patch_width] += 1
    
    # Average the overlapping regions
    reconstructed_volume = volume_accum / np.maximum(volume_count, 1)
    
    # Remove the padded region to obtain the original width.
    reconstructed_volume = reconstructed_volume[:, :, :original_width]
    
    return reconstructed_volume

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch image patches into a volume.")
    parser.add_argument("--patch_dir", required=True, help="Directory containing the patch .npy files.")
    parser.add_argument("--volume_file", required=True, help="Path to the original volume .npy file.")
    parser.add_argument("--save_path", required=True, help="Path to save the reconstructed volume as a .npy file.")
    parser.add_argument("--patch_height", type=int, default=254, help="Height of the patch (default: 254).")
    parser.add_argument("--patch_width", type=int, default=254, help="Width of the patch (default: 254).")
    parser.add_argument("--stride", type=int, default=50, help="Stride used during patch extraction (default: 50).")
    
    args = parser.parse_args()
    
    # Load the original volume and get its shape.
    original_volume = np.load(args.volume_file).transpose(0, 2, 1)
    num_slices, height, width = original_volume.shape
    
    # Stitch patches back into a volume.
    reconstructed = stitch_patches_to_volume(
        patch_dir=args.patch_dir,
        num_slices=num_slices,
        original_height=height,
        original_width=width,
        patch_size=(args.patch_height, args.patch_width),
        stride=args.stride
    )
    
    # Save the reconstructed volume.
    np.save(args.save_path, reconstructed)
    print(f"Reconstructed volume saved to {args.save_path}")
    