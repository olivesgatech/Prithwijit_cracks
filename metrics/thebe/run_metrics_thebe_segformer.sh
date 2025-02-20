#!/bin/bash
# run_metrics_cracks.sh
# Redirect all stdout and stderr to output.txt


OUTPUT_FILE="/home/prithwijit/Cracks/Prithwijit_cracks/metrics/output_cracks_testing_segformer.txt"

# Clear the output file if it exists


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks experts"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_expert/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_expert/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks experts"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_expert/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_expert/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks experts"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_expert/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_expert/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks prac"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_prac/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_prac/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks prac"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_prac/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_prac/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks prac"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_prac/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_prac/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks novice"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_novice/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_novice/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks novice"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_novice/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_novice/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks novice"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_novice/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_novice/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: thebe"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_thebe/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_thebe/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: thebe"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_thebe/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_thebe/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: thebe"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_thebe/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_thebe/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "pretrain_synth"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/pretrain_synth/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/pretrain_synth/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "pretrain_synth"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/pretrain_synth/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/pretrain_synth/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "pretrain_synth"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/pretrain_synth/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/pretrain_synth/pred_slices >> "$OUTPUT_FILE" 2>/dev/null