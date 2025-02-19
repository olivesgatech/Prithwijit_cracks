import os
import numpy as np
import argparse

def slice_volume(volume, output_folder):
    """
    Slices a 3D volume into 2D slices along the first axis and saves each slice.
    
    Parameters:
        volume (numpy.ndarray): The 3D volume of shape (400, 254, 701).
        output_folder (str): The folder to save the slices.
    """
    # Create output folder if it doesn't exist.
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over each slice in the volume.
    num_slices = volume.shape[0]
    for idx in range(num_slices):
        slice_array = volume[idx, :, :]
        # Build the file path using the naming convention.
        filename = os.path.join(output_folder, f"section_{idx}.npy")
        np.save(filename, slice_array)
        print(f"Saved slice {idx} to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slice a 3D numpy volume into individual 2D slices and save them."
    )
    parser.add_argument("volume_path", type=str,
                        help="Path to the input numpy volume file (.npy).")
    parser.add_argument("output_folder", type=str,
                        help="Path to the output folder where slices will be saved.")
    
    args = parser.parse_args()
    
    # Load the numpy volume from the given file.
    volume = np.load(args.volume_path)
    
    # Slice the volume and save the slices.
    slice_volume(volume, args.output_folder)
