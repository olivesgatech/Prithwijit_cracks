#!/bin/bash
# run_stitcher.sh

echo "Stitching predictions for segformer cracks"
python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_expert/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_expert/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_novice/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_novice/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_prac/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_prac/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_thebe/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_thebe/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/pretrain_synth/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/pretrain_synth/prediction.npy


echo "Stitching predictions for unet cracks"

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_expert/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_expert/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_novice/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_novice/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_prac/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_prac/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_thebe/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_thebe/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/pretrain_synth/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/pretrain_synth/prediction.npy


echo "Stitching predictions for unetpp cracks"

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_expert/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_expert/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_novice/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_novice/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_prac/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_prac/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_thebe/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_thebe/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/pretrain_synth/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/pretrain_synth/prediction.npy

echo "Stitching predictions for deeplab cracks"

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_expert/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_expert/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/prediction.npy

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/patch_stitcher.py \
    --patch_dir /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/logits \
    --volume_file "/mnt/HD_18T_pt1/prithwijit/CRACKS/image_volume.npy" \
    --save_path /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/prediction.npy