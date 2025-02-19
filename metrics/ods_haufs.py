import os
import glob
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, generate_binary_structure
from sklearn.neighbors import NearestNeighbors
import argparse
from skimage.metrics import hausdorff_distance

def calculate_modified_hausdorff_distance(true, pred):
    """
    Compute the modified Hausdorff distance between two binary masks.
    
    Parameters:
        pred_np (numpy.ndarray): Binary prediction mask.
        true_mask_np (numpy.ndarray): Binary ground truth mask.
        
    Returns:
        float: The modified Hausdorff distance between the two masks.
    """
    # Ensure the arrays are boolean
    pred_bool = pred.astype(bool)
    true_mask_bool = true.astype(bool)
    
    # Calculate and return the modified Hausdorff distance
    return hausdorff_distance(pred_bool, true_mask_bool, method='modified')

def main(args):
    # Folder paths for ground truth and prediction numpy files (for threshold selection)
    gt_folder = args.gt_folder
    pred_folder = args.pred_folder

    # List all ground truth files (sorted)
    gt_files = sorted(glob.glob(os.path.join(gt_folder, '*.npy')))

    # Build a dictionary for all prediction files keyed by their base filename
    pred_files_all = glob.glob(os.path.join(pred_folder, '*.npy'))
    pred_dict = {os.path.basename(f): f for f in pred_files_all}

    # Create a list of file pairs (ground truth, prediction)
    file_pairs = []
    for gt_file in gt_files:
        base = os.path.basename(gt_file)
        if base in pred_dict:
            file_pairs.append((gt_file, pred_dict[base]))
        else:
            print(f"Warning: No corresponding prediction file found for {gt_file}")

    print("Total file pairs for threshold selection:", len(file_pairs))

    # Loop over thresholds (from 0.01 to 0.99) and compute the metric
    metrics = []
    for th in tqdm(range(1, 100), desc="Processing thresholds"):
        metric_total = 0.0
        # Iterate over each pair of ground truth and prediction files
        for gt_file, pred_file in tqdm(file_pairs, desc=f"Threshold {th/100:.2f}", leave=False, total=len(file_pairs)):
            # Load the numpy arrays
            gt_image = np.load(gt_file)       # Binary ground truth image (0 or 1)
            pred_prob = np.load(pred_file)      # Probability map (already sigmoid-activated)

            # Apply threshold to prediction probabilities
            pred_binary = (pred_prob > (th / 100)).astype(np.uint8)

            # Skeletonize the thresholded prediction
            skeleton = skeletonize(pred_binary)
            # Dilate the skeleton to account for small misalignments
            structure = generate_binary_structure(2, 1)
            dilated = binary_dilation(skeleton, structure=structure, iterations=1)

            # Accumulate the metric using the Bidirectional Chamfer Distance
            metric_total += calculate_modified_hausdorff_distance(gt_image, dilated)

        # Average metric for the current threshold
        metrics.append(metric_total / len(file_pairs))

    # Convert the metrics list to a numpy array and determine the best threshold
    metrics = np.asarray(metrics)
    best_threshold = list(range(1, 100))[np.argmin(metrics)] / 100
    min_metric = np.min(metrics)

    print("Min Metric:", min_metric)
    print(f"Best Threshold: {best_threshold}")

    # Folder paths for evaluation predictions and ground truth files
    eval_pred_folder = args.eval_pred_folder
    eval_gt_folder = args.eval_gt_folder

    # Build a dictionary for evaluation ground truth files keyed by filename for fast lookup
    eval_gt_files = glob.glob(os.path.join(eval_gt_folder, '*.npy'))
    eval_gt_dict = {os.path.basename(f): f for f in eval_gt_files}

    # List evaluation prediction files and pair with corresponding ground truth files
    eval_pred_files = glob.glob(os.path.join(eval_pred_folder, '*.npy'))
    eval_file_pairs = []
    for pred_file in eval_pred_files:
        base = os.path.basename(pred_file)
        if base in eval_gt_dict:
            eval_file_pairs.append((eval_gt_dict[base], pred_file))
        else:
            print(f"Warning: No ground truth file for {pred_file}")

    print("Total evaluation file pairs:", len(eval_file_pairs))

    # Process each evaluation image using the best threshold
    total_metric = 0.0
    results = {}  # To store per-image metrics

    for gt_file, pred_file in tqdm(eval_file_pairs, desc="Evaluating predictions"):
        # Load ground truth (binary) and prediction (probability) images
        gt_image = np.load(gt_file)
        pred_prob = np.load(pred_file)

        # Threshold the prediction using the best threshold
        pred_binary = (pred_prob > best_threshold).astype(np.uint8)

        # Morphological processing: skeletonization and dilation
        skeleton = skeletonize(pred_binary)
        structure = generate_binary_structure(2, 1)
        dilated = binary_dilation(skeleton, structure=structure, iterations=1)

        # Compute the hausdorff_distance
        metric = calculate_modified_hausdorff_distance(gt_image, dilated)
        total_metric += metric

        # Store the metric with the image's filename for later inspection
        results[os.path.basename(pred_file)] = metric

    # Compute the average metric over all evaluated images
    if len(eval_file_pairs) > 0:
        avg_metric = total_metric / len(eval_file_pairs)
    else:
        avg_metric = None

    print("Evaluation Results:")
    print(f"Number of evaluated images: {len(eval_file_pairs)}")
    print(f"Average hausdorff_distance_2D: {avg_metric}")

    # Optionally, print per-image results
    for fname, m in results.items():
        print(f"{fname}: hausdorff_distance = {m}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run (hausdorff_distance) evaluation experiment."
    )
    parser.add_argument(
        "--gt_folder", type=str, required=True,
        help="Folder with binary ground truth .npy files (for threshold selection)"
    )
    parser.add_argument(
        "--pred_folder", type=str, required=True,
        help="Folder with prediction probability .npy files (sigmoid activated, for threshold selection)"
    )
    parser.add_argument(
        "--eval_gt_folder", type=str, required=True,
        help="Folder with evaluation ground truth .npy files"
    )
    parser.add_argument(
        "--eval_pred_folder", type=str, required=True,
        help="Folder with evaluation prediction .npy files (sigmoid activated)"
    )

    args = parser.parse_args()
    main(args)
