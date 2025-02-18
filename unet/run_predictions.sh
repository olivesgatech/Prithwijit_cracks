#!/bin/bash
# run_experiments.sh

# cracks testing

echo "Starting prediction on cracks finetuned model - synthetic"
python /home/prithwijit/Cracks/unet/predictions.py \
    --checkpoint_path "/home/prithwijit/Cracks/unet/checkpoints/best_unet_resnet50.pth" \
    --test_image_folder "/mnt/HD_18T_pt1/prithwijit/CRACKS/expert_cv_model/images" \
    --test_label_folder "/mnt/HD_18T_pt1/prithwijit/CRACKS/expert_cv_model/labels" \
    --prediction_logits_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/pretrain_synth/logits" \
    --save_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/pretrain_synth/examples"

echo "Starting prediction on cracks finetuned model - expert"
python /home/prithwijit/Cracks/unet/predictions.py \
    --checkpoint_path "/home/prithwijit/Cracks/unet/cracks_finetune/expert/best_model.pth" \
    --test_image_folder "/mnt/HD_18T_pt1/prithwijit/CRACKS/expert_cv_model/images" \
    --test_label_folder "/mnt/HD_18T_pt1/prithwijit/CRACKS/expert_cv_model/labels" \
    --prediction_logits_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_expert/logits" \
    --save_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_expert/examples"

echo "Starting prediction on cracks finetuned model - prac"
python /home/prithwijit/Cracks/unet/predictions.py \
    --checkpoint_path "/home/prithwijit/Cracks/unet/cracks_finetune/practitioner/best_model_latest.pth" \
    --test_image_folder "/mnt/HD_18T_pt1/prithwijit/CRACKS/expert_cv_model/images" \
    --test_label_folder "/mnt/HD_18T_pt1/prithwijit/CRACKS/expert_cv_model/labels" \
    --prediction_logits_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_prac/logits" \
    --save_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_prac/examples"


echo "Starting prediction on cracks finetuned model - novice"
python /home/prithwijit/Cracks/unet/predictions.py \
    --checkpoint_path "/home/prithwijit/Cracks/unet/cracks_finetune/novice/best_model.pth" \
    --test_image_folder "/mnt/HD_18T_pt1/prithwijit/CRACKS/expert_cv_model/images" \
    --test_label_folder "/mnt/HD_18T_pt1/prithwijit/CRACKS/expert_cv_model/labels" \
    --prediction_logits_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_novice/logits" \
    --save_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_novice/examples"

echo "Starting prediction on cracks finetuned model - thebe"
python /home/prithwijit/Cracks/unet/predictions.py \
    --checkpoint_path "/home/prithwijit/Cracks/unet/fine_thebe_dice_flat/best_unet_thebe_finetune.pth" \
    --test_image_folder "/mnt/HD_18T_pt1/prithwijit/CRACKS/expert_cv_model/images" \
    --test_label_folder "/mnt/HD_18T_pt1/prithwijit/CRACKS/expert_cv_model/labels" \
    --prediction_logits_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_thebe/logits" \
    --save_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/cracks/fine_thebe/examples"

# testing on synthetic

echo "Starting prediction on synthetic on pretrained model"
python /home/prithwijit/Cracks/unet/prediction_synthetic.py \
  --checkpoint_path "/home/prithwijit/Cracks/unet/checkpoints/best_unet_resnet50.pth" \
  --test_image_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/validation/seis" \
  --test_label_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/validation/fault" \
  --save_gt_folder "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/gt" \
  --save_pred_folder "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/logits" \
  --save_dir = "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/pretrain_synth/examples"

echo "Starting prediction on synthetic finetuned model - expert"
python /home/prithwijit/Cracks/unet/prediction_synthetic.py \
  --checkpoint_path "/home/prithwijit/Cracks/unet/cracks_finetune/expert/best_model.pth" \
  --test_image_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/validation/seis" \
  --test_label_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/validation/fault" \
  --save_gt_folder "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/gt" \
  --save_pred_folder "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/logits" \
  --save_dir = "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_expert/examples"

echo "Starting prediction on synthetic finetuned model - prac"
python /home/prithwijit/Cracks/unet/prediction_synthetic.py \
  --checkpoint_path "/home/prithwijit/Cracks/unet/cracks_finetune/practitioner/best_model_latest.pth" \
  --test_image_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/validation/seis" \
  --test_label_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/validation/fault" \
  --save_gt_folder "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/gt" \
  --save_pred_folder "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/logits" \
  --save_dir = "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_prac/examples"

echo "Starting prediction on synthetic finetuned model - novice"
python /home/prithwijit/Cracks/unet/prediction_synthetic.py \
  --checkpoint_path "/home/prithwijit/Cracks/unet/cracks_finetune/novice/best_model.pth" \
  --test_image_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/validation/seis" \
  --test_label_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/validation/fault" \
  --save_gt_folder "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/gt" \
  --save_pred_folder "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/logits" \
  --save_dir = "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_novice/examples"

echo "Starting prediction on synthetic finetuned model - thebe"
python /home/prithwijit/Cracks/unet/prediction_synthetic.py \
  --checkpoint_path "/home/prithwijit/Cracks/unet/fine_thebe_dice_flat/best_unet_thebe_finetune.pth" \
  --test_image_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/validation/seis" \
  --test_label_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/synth_data/FaultSeg3D/validation/fault" \
  --save_gt_folder "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/gt" \
  --save_pred_folder "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/logits" \
  --save_dir = "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/synthetic/fine_thebe/examples"



#thebe testing 

echo "Starting prediction on thebe on pretrained model"
python /home/prithwijit/Cracks/unet/predictions_thebe.py \
    --checkpoint_path "/home/prithwijit/Cracks/unet/checkpoints/best_unet_resnet50.pth" \
    --test_image_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/Thebe/test/seismic" \
    --test_label_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/Thebe/test/labels" \
    --prediction_logits_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/thebe/pretrain_synth/logits" \
    --save_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/thebe/pretrain_synth/logits/examples"

echo "Starting prediction on thebe on finetune expert"
python /home/prithwijit/Cracks/unet/predictions_thebe.py \
    --checkpoint_path "/home/prithwijit/Cracks/unet/cracks_finetune/expert/best_model.pth" \
    --test_image_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/Thebe/test/seismic" \
    --test_label_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/Thebe/test/labels" \
    --prediction_logits_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/thebe/fine_expert/logits" \
    --save_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/thebe/fine_expert/examples"

echo "Starting prediction on thebe on finetune prac"
python /home/prithwijit/Cracks/unet/predictions_thebe.py \
    --checkpoint_path "/home/prithwijit/Cracks/unet/cracks_finetune/practitioner/best_model_latest.pth" \
    --test_image_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/Thebe/test/seismic" \
    --test_label_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/Thebe/test/labels" \
    --prediction_logits_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/thebe/fine_prac/logits" \
    --save_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/thebe/fine_prac/examples"

echo "Starting prediction on thebe on finetune novice"
python /home/prithwijit/Cracks/unet/predictions_thebe.py \
    --checkpoint_path "/home/prithwijit/Cracks/unet/cracks_finetune/novice/best_model.pth" \
    --test_image_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/Thebe/test/seismic" \
    --test_label_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/Thebe/test/labels" \
    --prediction_logits_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/thebe/fine_novice/logits" \
    --save_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/thebe/fine_novice/examples"

echo "Starting prediction on thebe on finetune thebe"
python /home/prithwijit/Cracks/unet/predictions_thebe.py \
    --checkpoint_path "/home/prithwijit/Cracks/unet/fine_thebe_dice_flat/best_unet_thebe_finetune.pth" \
    --test_image_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/Thebe/test/seismic" \
    --test_label_folder "/mnt/HD_18T_pt1/prithwijit/Faultseg/Thebe/test/labels" \
    --prediction_logits_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/thebe/fine_thebe/logits" \
    --save_dir "/mnt/HD_18T_pt1/prithwijit/models_predictions/unet/thebe/fine_thebe/examples"