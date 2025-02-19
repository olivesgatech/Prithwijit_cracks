#!/bin/bash
# run_metrics_cracks.sh
# Redirect all stdout and stderr to output.txt

exec >> /home/prithwijit/Cracks/Prithwijit_cracks/metrics/output_cracks_testing.txt 2>&1


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "finetune: cracks experts"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_expert/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_expert/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "finetune: cracks experts"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_expert/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_expert/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "finetune: cracks experts"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_expert/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_expert/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "finetune: cracks prac"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "finetune: cracks prac"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "finetune: cracks prac"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "finetune: cracks novice"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "finetune: cracks novice"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "finetune: cracks novice"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "finetune: thebe"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "finetune: thebe"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "finetune: thebe"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "pretrain_synth"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_bcd.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "pretrain_synth"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_dice.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices


echo "Evaluation for deeplab"
echo "test set: cracks"
echo "pretrain_synth"

python /home/prithwijit/Cracks/Prithwijit_cracks/metrics/ods_haufs.py --gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/train \
                   --pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices \
                   --eval_gt_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/gt/test \
                   --eval_pred_folder /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices