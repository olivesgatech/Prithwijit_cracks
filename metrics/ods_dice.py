import os
import glob
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, generate_binary_structure
import argparse

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

def main(args):
    # --- Threshold Selection on the first dataset ---
    gt_folder = args.gt_folder
    pred_folder = args.pred_folder

    # List all ground truth files (sorted)
    gt_files = sorted(glob.glob(os.path.join(gt_folder, '*.npy')))
    
    # Build a dictionary for prediction files keyed by filename
    pred_files_all = glob.glob(os.path.join(pred_folder, '*.npy'))
    pred_dict = {os.path.basename(f): f for f in pred_files_all}
    
    # Create list of file pairs (ground truth, prediction)
    file_pairs = []
    for gt_file in gt_files:
        base = os.path.basename(gt_file)
        if base in pred_dict:
            file_pairs.append((gt_file, pred_dict[base]))
        else:
            print(f"Warning: No corresponding prediction file found for {gt_file}")
    
    print("Total file pairs for threshold selection:", len(file_pairs))
    
    # Loop over thresholds (from 0.01 to 0.99) and compute average Dice score
    dice_scores = []
    for th in tqdm(range(1, 100), desc="Processing thresholds"):
        dice_total = 0.0
        for gt_file, pred_file in tqdm(file_pairs, desc=f"Threshold {th/100:.2f}", leave=False, total=len(file_pairs)):
            # Load the numpy arrays
            gt_image = np.load(gt_file)       # Binary ground truth image (0 or 1)
            pred_prob = np.load(pred_file)      # Probability map (sigmoid activated)
            
            # Apply threshold to prediction probabilities
            pred_binary = (pred_prob > (th / 100)).astype(np.uint8)
            
            # Optional: Uncomment these lines if you wish to use skeletonization and dilation:
            skeleton = skeletonize(pred_binary)
            structure = generate_binary_structure(2, 1)
            pred_binary = binary_dilation(skeleton, structure=structure, iterations=1)
            
            # Compute Dice score
            dice = compute_dice_score(gt_image, pred_binary)
            dice_total += dice
        
        # Average Dice for the current threshold
        dice_scores.append(dice_total / len(file_pairs))
    
    dice_scores = np.asarray(dice_scores)
    # For Dice score, best threshold is the one with the maximum score.
    best_threshold = list(range(1, 100))[np.argmax(dice_scores)] / 100
    max_dice = np.max(dice_scores)
    
    print("Max Average Dice Score:", max_dice)
    print(f"Best Threshold: {best_threshold}")
    
    # --- Evaluation on the evaluation dataset ---
    eval_gt_folder = args.eval_gt_folder
    eval_pred_folder = args.eval_pred_folder
    
    # Build dictionary for evaluation ground truth files keyed by filename
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
    
    total_dice = 0.0
    results = {}  # To store per-image Dice scores
    for gt_file, pred_file in tqdm(eval_file_pairs, desc="Evaluating predictions"):
        gt_image = np.load(gt_file)
        pred_prob = np.load(pred_file)
        
        # Apply the best threshold
        pred_binary = (pred_prob > best_threshold).astype(np.uint8)
        
        # Optional: Uncomment if you wish to apply post-processing:
        skeleton = skeletonize(pred_binary)
        structure = generate_binary_structure(2, 1)
        pred_binary = binary_dilation(skeleton, structure=structure, iterations=1)
        
        dice = compute_dice_score(gt_image, pred_binary)
        total_dice += dice
        
        results[os.path.basename(pred_file)] = dice
    
    if len(eval_file_pairs) > 0:
        avg_dice = total_dice / len(eval_file_pairs)
    else:
        avg_dice = None
    
    print("Evaluation Results:")
    print(f"Number of evaluated images: {len(eval_file_pairs)}")
    print(f"Average Dice Score: {avg_dice}")
    
    # Optionally, print per-image Dice scores
    for fname, score in results.items():
        print(f"{fname}: Dice Score = {score}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run Dice Score evaluation experiment."
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
