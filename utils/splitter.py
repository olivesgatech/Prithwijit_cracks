import os
import glob
import shutil
import re

# Set the path to the folder containing your numpy files
source_folder = "/mnt/HD_18T_pt1/prithwijit/models_predictions/gt"  # update this to your folder path

# Define train and test folders
train_folder = os.path.join(source_folder, "train")
test_folder = os.path.join(source_folder, "test")

# Create train and test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get a list of all .npy files in the source folder
npy_files = glob.glob(os.path.join(source_folder, "*.npy"))

# Process each file
for file_path in npy_files:
    filename = os.path.basename(file_path)  # e.g., "section_23.npy"
    # Extract the index from the filename using a regex
    match = re.search(r'_(\d+)\.npy', filename)
    if match:
        index = int(match.group(1))
        # If index is in [0, 29] or [370, 399], move to test folder; else, to train folder.
        if index < 30 or index >= 370:
            dest = os.path.join(test_folder, filename)
        else:
            dest = os.path.join(train_folder, filename)
        shutil.move(file_path, dest)
    else:
        print(f"Skipping file with unexpected naming convention: {filename}")
