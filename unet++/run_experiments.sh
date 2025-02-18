#!/bin/bash
# run_experiments.sh

# Experiment 1
# echo "Starting Experiment 1..."
# python /home/prithwijit/Cracks/unetpp/unetpp_finetuning.py --image_folder /mnt/HD_18T_pt1/prithwijit/CRACKS/expert_cv_model/train/images \
#                           --label_folder /mnt/HD_18T_pt1/prithwijit/CRACKS/expert_cv_model/train/labels \
#                           --resume_checkpoint /home/prithwijit/Cracks/unetpp/checkpoints/bestpp_unet_resnet50.pth \
#                           --best_checkpoint /home/prithwijit/Cracks/unetpp/cracks_finetune/expert/best_model.pth \
#                           --current_checkpoint /home/prithwijit/Cracks/unetpp/cracks_finetune/expert/current_checkpoint.pth \
#                           --project "unetpp-expert"

# Experiment 2
echo "Starting Experiment 2..."
python /home/prithwijit/Cracks/unetpp/unetpp_finetuning.py --image_folder /mnt/HD_18T_pt1/prithwijit/CRACKS/prac_cv_model/train/images \
                          --label_folder /mnt/HD_18T_pt1/prithwijit/CRACKS/prac_cv_model/train/labels \
                          --resume_checkpoint /home/prithwijit/Cracks/unetpp/checkpoints/bestpp_unet_resnet50.pth \
                          --best_checkpoint /home/prithwijit/Cracks/unetpp/cracks_finetune/practitioner/best_model_latest.pth \
                          --current_checkpoint /home/prithwijit/Cracks/unetpp/cracks_finetune/practitioner/current_checkpoint.pth \
                          --project "unetpp-practitioner"

# Experiment 3
echo "Starting Experiment 3..."
python /home/prithwijit/Cracks/unetpp/unetpp_finetuning.py --image_folder /mnt/HD_18T_pt1/prithwijit/CRACKS/novice_cv_model/train/images \
                          --label_folder /mnt/HD_18T_pt1/prithwijit/CRACKS/novice_cv_model/train/labels \
                          --resume_checkpoint /home/prithwijit/Cracks/unetpp/checkpoints/bestpp_unet_resnet50.pth \
                          --best_checkpoint /home/prithwijit/Cracks/unetpp/cracks_finetune/novice/best_model.pth \
                          --current_checkpoint /home/prithwijit/Cracks/unetpp/cracks_finetune/novice/current_checkpoint.pth \
                          --project "unetpp-novice"

echo "All experiments completed."
