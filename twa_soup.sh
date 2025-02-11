#!/bin/bash

model=clip-vit-b32
dataset=ImageNet
DST=../models/
wd=0
seed=0
devices=0
val_ratio=2
port=1234
lr=0.001
bz=32
bits=4
layer=0

CUDA_VISIBLE_DEVICES=$devices python -m torch.distributed.launch --nproc_per_node 1 --master_port $port train_twa.py \
    --lr $lr --batch-size $bz --wd $wd --epochs 5 \
    --cosine-lr --optimizer adamw --layer-size $layer \
    --model-location $DST \
    --datasets $dataset --params_start 0 --params_end 72 --arch $model --val_ratio $val_ratio \
    --optimizer adamw --split --ddp --rho 0 --randomseed $seed \
    --finetune --model_batch 72 --models_epochs 1 --Pbits $bits
