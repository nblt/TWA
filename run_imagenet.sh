#!/bin/bash

# basic training 
seed=1
datasets=ImageNet
port=1234
model=vit-b-16
opt=adamw
val_ratio=2
epochs=90
DST=results_sgd/no_shuffle_$opt\_epoch$epochs\_$model\_$datasets\_split$val_ratio\_seed$seed
CUDA_VISIBLE_DEVICES=$device python train_sgd_cifar.py --datasets $datasets \
        --arch=$model --epochs=$epochs --lr 0.001 --optimizer $opt  --wd 0.1 --schedule cosine \
        --save-dir=$DST/checkpoints --log-dir=$DST -p 100 --split -b 128 --workers 4 --val_ratio $val_ratio \
        --randomseed $seed

# TWA training
# CIFAR

lr=0.01
wd=0
val_ratio=2
opt=adamw
epochs=90
DST=/opt/data/private/litao/TWA_arch/TWA/results_sgd/$model\_$datasets\_split$val_ratio\_seed$seed/checkpoints
bits=4

CUDA_VISIBLE_DEVICES=$devices python -m torch.distributed.launch --nproc_per_node 1 --master_port $port train_twa.py \
    --lr $lr --batch-size $batch --wd $wd --epochs 10 \
    --cosine-lr --optimizer adamw --layer-size $layer \
    --model-location $DST \
    --datasets $datasets --params_start 1 --params_end  100 --arch $model --val_ratio $val_ratio \
    --optimizer adamw --split --ddp --rho 0 --randomseed $seed \
    --model_batch 100 --models_epochs 1 --Pbits $bits

