#!/bin/bash

# CIFAR experiments

datasets=CIFAR100
device=0

####################################### swa ########################################

seed=0
swa_lr=0.05 # 0.05 / 0.10
model=PreResNet164
wd_psgd=0.00005
DST=swa_$model\_$datasets\_$seed\_$swa_lr

CUDA_VISIBLE_DEVICES=$device python -u train_twa.py --epochs 10 --datasets $datasets \
        --opt SGD --extract Schmidt --schedule step --accumulate 1 \
        --lr 2 --params_start 126 --params_end 226 --train_start 225 --wd $wd_psgd \
        --batch-size 128  --arch=$model  \
        --save-dir=$DST/checkpoints  --log-dir=$DST --log-name=from_last

seed=0
swa_lr=0.05 # 0.05 / 0.10
model=VGG16BN
wd_psgd=0.00005
DST=swa_$model\_$datasets\_$seed\_$swa_lr

CUDA_VISIBLE_DEVICES=$device python -u train_twa.py --epochs 10 --datasets $datasets \
        --opt SGD --extract Schmidt --schedule step --accumulate 1 \
        --lr 2 --params_start 161 --params_end 301 --train_start 300 --wd $wd_psgd \
        --batch-size 128  --arch=$model  \
        --save-dir=$DST/checkpoints  --log-dir=$DST --log-name=from_last