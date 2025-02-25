#!/bin/bash

# Define base directory
BASE_DIR="/mnt/HD_18T_pt1/prithwijit/models_predictions/unet"
MODELS=("unet" "unetpp" "deeplab" "segformer")
CATEGORIES=("fine_expert" "fine_prac" "fine_novice" "fine_thebe" "pretrain_synth")

# Loop through models and categories
for MODEL in "${MODELS[@]}"; do
    for CATEGORY in "${CATEGORIES[@]}"; do
        INPUT_PATH="/mnt/HD_18T_pt1/prithwijit/models_predictions/${MODEL}/thebe/${CATEGORY}/logits_train/"
        OUTPUT_PATH="/mnt/HD_18T_pt1/prithwijit/models_predictions/${MODEL}/thebe/${CATEGORY}/logits_train_stitched/"
        
        echo "Processing: $MODEL - $CATEGORY"
        python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher_thebe.py "$INPUT_PATH" "$OUTPUT_PATH"
    done
done
