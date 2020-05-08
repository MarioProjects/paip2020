#!/bin/bash

# pspnet_resnetd101b_scratch, pspnet_resnetd101b_imagenet_encoder,
# pspnet_resnetd101b_coco, pspnet_resnetd101b_coco_encoder
# pspnet_resnetd101b_voc, pspnet_resnetd101b_voc_encoder,
# resnet_unet_scratch, resnet34_unet_scratch, resnet34_unet_imagenet_encoder
model="small_segmentation_unet"
gpu="0,1"
seed=2020
train_mode="low_resolution"

#data_fold=0 # 0 - 1 - 2 - 3 - 4

slide_level=2
low_res=2048 # 512, 1024, 1504, 2048
img_size=2048
crop_size=2048

epochs=20
defrost_epoch=-1
batch_size=2

optimizer="adam"     # adam - over9000
scheduler="constant" # one_cycle_lr
min_lr=0.0001
max_lr=0.01
#lr=0.0001 # learning_rate for conventional schedulers

# bce_dice - bce_dice_ac - bce_dice_border
criterion="bce_dice"
weights_criterion="0.1,0.85,0.05"

for data_augmentation in "none" "combination"
do
# "none" - "rotations" - "vflips" - "hflips" - "elastic_transform" - "grid_distortion"
# "shift" - "scale" - "optical_distortion" - "coarse_dropout" - "random_crops" - "downscale"
#data_augmentation="none"

for data_fold in 0 1 2 3 4
do

for lr in 0.01 0.001 0.0001
do

parent_dir="lvl${slide_level}_lowres${low_res}/$model/$optimizer/${criterion}_weights${weights_criterion}"
model_path="results/$parent_dir/datafold${data_fold}_da${data_augmentation}_minlr${min_lr}_maxlr${max_lr}_lr${lr}"

echo -e "\n---- Start Initial Training ----\n"
# --normalize
python3 -u train.py --gpu $gpu --output_dir $model_path --epochs $epochs --defrost_epoch $defrost_epoch \
  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation \
  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer \
  --slide_level $slide_level --low_res $low_res --training_mode $train_mode \
  --criterion $criterion --weights_criterion $weights_criterion \
  --min_lr $min_lr --max_lr $max_lr --seed $seed --scheduler_steps 99 --learning_rate $lr \
  --data_fold $data_fold

#echo -e "\n---- Apply Stochastic Weight Averaging (SWA) ----\n"

#swa_start=0
#swa_freq=1
#swa_lr=0.001
#swa_epochs=50
#initial_checkpoint="${model_path}model_${model}_best_iou.pt"
#optimizer="sgd"
#scheduler="constant"
#python3 -u train.py --gpu $gpu --output_dir $model_path --epochs $swa_epochs --defrost_epoch $defrost_epoch \
#  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation \
#  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer \
#  --slide_level $slide_level --low_res $low_res --training_mode $train_mode \
#  --criterion $criterion --weights_criterion $weights_criterion \
#  --apply_swa --swa_start $swa_start --swa_freq $swa_freq --swa_lr $swa_lr \
#  --model_checkpoint $initial_checkpoint --seed $seed

done

done

done



python3 utils/slack_message.py --msg "[PAIP 2020] Experiment finished!"
