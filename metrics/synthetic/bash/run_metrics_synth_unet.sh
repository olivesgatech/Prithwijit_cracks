#!/bin/bash
# run_metrics_cracks.sh
# Redirect all stdout and stderr to output.txt


OUTPUT_FILE="/home/prithwijit/Cracks/Prithwijit_cracks/metrics/synthetic/output_synth_testing_unet.txt"

# Clear the output file if it exists


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "finetune: cracks experts"  >> "$OUTPUT_FILE"

#/home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "finetune: cracks experts"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "finetune: cracks experts"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "finetune: cracks prac"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "finetune: cracks prac"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "finetune: cracks prac"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "finetune: cracks novice"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "finetune: cracks novice"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "finetune: cracks novice"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "finetune: thebe"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "finetune: thebe"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "finetune: thebe"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "pretrain_synth"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "pretrain_synth"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/logits >> "$OUTPUT_FILE" 2>/dev/null


echo "Evaluation for unet"  >> "$OUTPUT_FILE"
echo "test set: synth"  >> "$OUTPUT_FILE"
echo "pretrain_synth"  >> "$OUTPUT_FILE"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/thebe/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/gt_train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/logits_train \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/gt  \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/logits >> "$OUTPUT_FILE" 2>/dev/null
