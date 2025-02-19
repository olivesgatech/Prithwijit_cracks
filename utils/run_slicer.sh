#!/bin/bash
# run_slicer.sh

# python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py /mnt/HD_18T_pt1/prithwijit/CRACKS/experts/ATHOQ52EXZNQ9_npy_morphed.npy /mnt/HD_18T_pt1/prithwijit/models_predictions


echo "deeplab"


python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_expert/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_expert/pred_slices


python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_novice/pred_slices


python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_prac/pred_slices

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/fine_thebe/pred_slices

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/deeplab/cracks/pretrain_synth/pred_slices

echo "segformer"

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_expert/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_expert/pred_slices


python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_novice/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_novice/pred_slices


python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_prac/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_prac/pred_slices

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_thebe/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/fine_thebe/pred_slices

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/pretrain_synth/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/segformer/cracks/pretrain_synth/pred_slices

echo "unet"

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_expert/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_expert/pred_slices


python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_novice/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_novice/pred_slices


python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_prac/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_prac/pred_slices

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_thebe/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_thebe/pred_slices

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/pretrain_synth/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/pretrain_synth/pred_slices


echo "unetpp"

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_expert/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_expert/pred_slices


python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_novice/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_novice/pred_slices


python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_prac/prediction.npy \
    -/mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_prac/pred_slices

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_thebe/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/fine_thebe/pred_slices

python /home/prithwijit/Cracks/Prithwijit_cracks/utils/slicer.py \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/pretrain_synth/prediction.npy \
    /mnt/HD_18T_pt1/prithwijit/models_predictions/unetpp/cracks/pretrain_synth/pred_slices