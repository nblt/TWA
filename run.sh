#!/bin/bash

################################ CIFAR ###################################
datasets=CIFAR100
device=0
model=VGG16BN # PreResNet164
DST=results/$model\_$datasets\_seed$seed

CUDA_VISIBLE_DEVICES=$device python -u train_sgd_cifar.py --datasets $datasets \
        --arch=$model --epochs=200 --wd=$wd --randomseed $seed --lr 0.1 \
        --save-dir=$DST/checkpoints --log-dir=$DST 

lr=2
end=101
wd_psgd=0.00001
CUDA_VISIBLE_DEVICES=$device python -u train_twa.py --epochs 10 --datasets $datasets \
        --opt SGD --extract Schmidt --schedule step \
        --lr $lr --params_start 0 --params_end $end --train_start -1 --wd $wd_psgd \
        --batch-size 128  --arch=$model  \
        --save-dir=$DST/checkpoints  --log-dir=$DST


################################ ImageNet ################################
datasets=ImageNet
device=0,1,2,3

model=resnet18
path=/home/datasets/ILSVRC2012/
CUDA_VISIBLE_DEVICES=$device  python3 train_sgd_imagenet.py -a $model \
    --epochs 90 --workers 8  --dist-url 'tcp://127.0.0.1:1234' \
    --dist-backend 'nccl' --multiprocessing-distributed \
    --world-size 1 --rank 0 $path

# TWA 60+2
wd_psgd=0.00001
lr=0.3
DST=save_resnet18
CUDA_VISIBLE_DEVICES=$device python -u train_twa.py --epochs 2 --datasets $datasets \
        --opt SGD --extract Schmidt --schedule step --worker 8 \
        --lr $lr --params_start 0 --params_end 301 --train_start -1 --wd $wd_psgd \
        --batch-size 256  --arch=$model  \
        --save-dir=$DST  --log-dir=$DST 

# TWA 90+1
wd_psgd=0.00001
lr=0.03
DST=save_resnet18
CUDA_VISIBLE_DEVICES=$device python -u train_twa.py --epochs 1 --datasets $datasets \
        --opt SGD --extract Schmidt --schedule linear --worker 8 \
        --lr $lr --params_start 301 --params_end 451 --train_start -1 --wd $wd_psgd \
        --batch-size 256  --arch=$model  \
        --save-dir=$DST  --log-dir=$DST 
