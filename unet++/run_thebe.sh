#!/bin/bash
# run_experiments.sh

# Experiment 1
echo "Starting Experiment 1..."
python /home/prithwijit/Cracks/unetpp/unetpp_finetune_thebe.pyc --image_folder /mnt/HD_18T_pt1/prithwijit/Faultseg/Thebe/seismic \
                          --label_folder /mnt/HD_18T_pt1/prithwijit/Faultseg/Thebe/labels \
                          --resume_checkpoint /home/prithwijit/Cracks/unetpp/checkpoints/bestpp_unet_resnet50.pth \
                          --best_checkpoint /home/prithwijit/Cracks/unetpp/fine_thebe_dice_flat/best_model.pth \
                          --current_checkpoint /home/prithwijit/Cracks/unetpp/fine_thebe_dice_flat/current_checkpoint.pth \
                          --project "unetpp-thebe"


echo "All experiments completed."
