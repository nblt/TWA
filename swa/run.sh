device=0
data_dir=../datasets/

############################### VGG16 ###################################
dataset=CIFAR100
model=VGG16BN
seed=0
swa_lr=0.05
dir=swa_$model\_$dataset\_$seed\_$swa_lr
UDA_VISIBLE_DEVICES=$device python3 train.py --dir=$dir --dataset=$dataset --data_path=$data_dir \
    --model=$model --epochs=300 --lr_init=0.1 --wd=5e-4 --seed $seed \
    --swa --swa_start=161 --swa_lr=$swa_lr |& tee -a $dir/log # SWA 1.5 Budgets


############################### PreResNet ###################################
dataset=CIFAR100 # CIFAR10 CIFAR100
model=PreResNet164
seed=0
swa_lr=0.05

dir=swa_$model\_$dataset\_$seed\_$swa_lr
CUDA_VISIBLE_DEVICES=$device python3 train.py --dir=$dir --seed $seed\
    --dataset=$dataset --data_path=$data_dir  --model=PreResNet164 --epochs=225 \
    --lr_init=0.1 --wd=3e-4 --swa --swa_start=126 --swa_lr=$swa_lr |& tee -a $dir/log # SWA 1.5 Budgets

