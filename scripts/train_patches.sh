#!/bin/bash

# pspnet_resnetd101b_scratch, pspnet_resnetd101b_imagenet_encoder,
# pspnet_resnetd101b_coco, pspnet_resnetd101b_coco_encoder
# pspnet_resnetd101b_voc, pspnet_resnetd101b_voc_encoder,
# resnet_unet_scratch, resnet34_unet_scratch, resnet34_unet_imagenet_encoder
model="resnet34_unet_imagenet_encoder"
gpu="0,1"
seed=2020
train_mode="patches"

slide_level=2
patch_len=256
stride_len=64

epochs=30
defrost_epoch=7
batch_size=24
samples_per_type=20000

optimizer="adam"     # adam - over9000
scheduler="constant" # one_cycle_lr
min_lr=0.0001
max_lr=0.01
lr=0.01 # learning_rate for conventional schedulers

# bce_dice - bce_dice_ac - bce_dice_border
criterion="bce_dice"
weights_criterion="0.1,0.85,0.05"

img_size=256
crop_size=256

# "none" - "rotations" - "flips" - "elastic_transform" - "grid_distortion"
# "shift" - "scale" - "optical_distortion" - "coarse_dropout"
data_augmentation="none"

parent_dir="lvl${slide_level}_patch${patch_len}_stride$stride_len/samples_$samples_per_type/$model/$optimizer/$criterion"
model_path="results/$parent_dir/weights${weights_criterion}_da${data_augmentation}_minlr${min_lr}_maxlr${max_lr}"

echo -e "\n---- Start Initial Training ----\n"
# --normalize
python3 -u train.py --gpu $gpu --output_dir $model_path --epochs $epochs --defrost_epoch $defrost_epoch \
  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation \
  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer \
  --slide_level $slide_level --patch_len $patch_len --stride_len $stride_len --training_mode $train_mode \
  --samples_per_type $samples_per_type --criterion $criterion --weights_criterion $weights_criterion \
  --min_lr $min_lr --max_lr $max_lr --seed $seed --scheduler_steps 99

echo -e "\n---- Apply Stochastic Weight Averaging (SWA) ----\n"

swa_start=0
swa_freq=1
swa_lr=0.001
swa_epochs=50
initial_checkpoint="${model_path}model_${model}_best_iou.pt"
optimizer="sgd"
scheduler="constant"
python3 -u train.py --gpu $gpu --output_dir $model_path --epochs $swa_epochs --defrost_epoch $defrost_epoch \
  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation \
  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer \
  --slide_level $slide_level --patch_len $patch_len --stride_len $stride_len --training_mode $train_mode \
  --samples_per_type $samples_per_type --criterion $criterion --weights_criterion $weights_criterion \
  --apply_swa --swa_start $swa_start --swa_freq $swa_freq --swa_lr $swa_lr \
  --model_checkpoint $initial_checkpoint --seed $seed

python3 utils/slack_message.py --msg "[PAIP 2020] Experiment finished!"
