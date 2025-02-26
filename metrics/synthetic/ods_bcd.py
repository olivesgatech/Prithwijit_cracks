import os
import glob
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, generate_binary_structure
from sklearn.neighbors import NearestNeighbors
import argparse
from skimage.metrics import hausdorff_distance

def compute_dice_score(gt, pred):
    """
    Compute the Dice score between two binary masks.
    If both masks are empty, returns 1.0.
    """
    intersection = np.sum(gt * pred)
    gt_sum = np.sum(gt)
    pred_sum = np.sum(pred)
    
    if gt_sum + pred_sum == 0:
        return 1.0  # perfect match: both masks are empty
    return 2 * intersection / (gt_sum + pred_sum)

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
    max_distance = np.sqrt(true.shape[0]**2 + true.shape[1]**2)
    pred_bool = pred.astype(bool)
    true_mask_bool = true.astype(bool)
    
    # Calculate and return the modified Hausdorff distance
    dist = hausdorff_distance(pred_bool, true_mask_bool, method='modified')
    return max_distance if dist == float('inf') else dist
    


def compute_bcd_2d(label, prediction):
    max_distance = np.sqrt(label.shape[0]**2 + label.shape[1]**2)
    label_points = np.argwhere(label == 1)
    prediction_points = np.argwhere(prediction == 1)

    if len(label_points) == 0 and len(prediction_points) == 0:
        return 0
    if len(label_points) == 0 or len(prediction_points) == 0:
        return np.inf

    nbrs_label = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(prediction_points)
    dists_label_to_pred, _ = nbrs_label.kneighbors(label_points)

    nbrs_pred = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(label_points)
    dists_pred_to_label, _ = nbrs_pred.kneighbors(prediction_points)

    min_dist_label_to_pred = np.mean(dists_label_to_pred)
    min_dist_pred_to_label = np.mean(dists_pred_to_label)

    dist = min_dist_label_to_pred + min_dist_pred_to_label
    return max_distance if dist == float('inf') else dist           

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
            # print(f"Warning: No corresponding prediction file found for {gt_file}")
            x = 1

    # print("Total file pairs for threshold selection:", len(file_pairs))

    # Loop over thresholds (from 0.01 to 0.99) and compute the metric
    thresholds = np.arange(1, 100, 5)
    metrics = []
    for th in tqdm(thresholds, desc="Processing thresholds"):
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
            metric_total += compute_bcd_2d(gt_image, dilated)

        # Average metric for the current threshold
        metrics.append(metric_total / len(file_pairs))

    # Convert the metrics list to a numpy array and determine the best threshold
    metrics = np.asarray(metrics)
    best_threshold = thresholds[np.argmin(metrics)] / 100
    min_metric = np.min(metrics)

    # print("Min Metric:", min_metric)
    print(f"Best Threshold for BCD: {best_threshold}")

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
            # print(f"Warning: No ground truth file for {pred_file}")
            x =1

    # print("Total evaluation file pairs:", len(eval_file_pairs))

    # Process each evaluation image using the best threshold
    total_metric = 0.0
    total_metric1 = 0.0
    total_metric2 = 0.0
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

        # Compute the Bidirectional Chamfer Distance (BCD)
        metric = compute_bcd_2d(gt_image, dilated)
        metric1 = compute_dice_score(gt_image, dilated)
        metric2 = calculate_modified_hausdorff_distance(gt_image, dilated)
        total_metric += metric
        total_metric1 += metric1
        total_metric2 += metric2

        # Store the metric with the image's filename for later inspection
        # results[os.path.basename(pred_file)] = metric

    # Compute the average metric over all evaluated images
    if len(eval_file_pairs) > 0:
        avg_metric = total_metric / len(eval_file_pairs)
    else:
        avg_metric = None
    if len(eval_file_pairs) > 0:
        avg_metric1 = total_metric1 / len(eval_file_pairs)
    else:
        avg_metric1 = None
    if len(eval_file_pairs) > 0:
        avg_metric2 = total_metric2 / len(eval_file_pairs)
    else:
        avg_metric2 = None


    print(f"Average BCD_2D on BCD Threshold: {avg_metric}")
    print(f"Average DICE on BCD Threshold: {avg_metric1}")
    print(f"Average Hausdorff on BCD Threshold: {avg_metric2}")
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run Bidirectional Chamfer Distance (BCD_2D) evaluation experiment."
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
