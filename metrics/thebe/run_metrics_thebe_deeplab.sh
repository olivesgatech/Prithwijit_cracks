#!/bin/bash
# run_metrics_cracks.sh
# Redirect all stdout and stderr to output.txt


OUTPUT_FILE="/home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/output_thebe_testing_deeplab.txt"

# Clear the output file if it exists


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks experts"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/thebe/fine_expert/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/thebe/fine_expert/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks experts"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/thebe/fine_expert/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/thebe/fine_expert/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks experts"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/thebe/fine_expert/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/thebe/fine_expert/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks prac"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks prac"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks prac"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks novice"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks novice"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: cracks novice"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: thebe"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: thebe"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "finetune: thebe"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "pretrain_synth"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "pretrain_synth"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for deeplab"  >> "$OUTPUT_FILE"
echo "test set: cracks"  >> "$OUTPUT_FILE"
echo "pretrain_synth"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices >> "$OUTPUT_FILE" 2>/dev/null