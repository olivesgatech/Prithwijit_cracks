#!/bin/bash
# run_metrics_cracks.sh
# Redirect all stdout and stderr to output.txt


OUTPUT_FILE="/home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/output_thebe_testing_segformer.txt"

# Clear the output file if it exists


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "finetune: cracks experts"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_expert/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_expert/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "finetune: cracks experts"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_expert/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_expert/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "finetune: cracks experts"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_expert/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_expert/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "finetune: cracks prac"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_prac/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_prac/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "finetune: cracks prac"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_prac/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_prac/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "finetune: cracks prac"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_prac/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_prac/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "finetune: cracks novice"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_novice/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_novice/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "finetune: cracks novice"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_novice/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_novice/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "finetune: cracks novice"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_novice/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_novice/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "finetune: thebe"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_thebe/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_thebe/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "finetune: thebe"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_thebe/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_thebe/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "finetune: thebe"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_thebe/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/fine_thebe/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "pretrain_synth"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/pretrain_synth/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/pretrain_synth/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "pretrain_synth"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/pretrain_synth/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/pretrain_synth/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for segformer"  >> "$OUTPUT_FILE"
echo "test set: thebe"  >> "$OUTPUT_FILE"
echo "pretrain_synth"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/pretrain_synth/logits_train_stitched \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt_thebe-test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/thebe/pretrain_synth/logits_stitched >> "$OUTPUT_FILE" 2>/dev/null