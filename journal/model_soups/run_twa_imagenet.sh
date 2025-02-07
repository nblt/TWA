wd=0.0000
devices=0,1
port=1234
datasets=ImageNet
DST=/home/dev/imagenet/

for block in 6
do
    for lr in 0.3
    do
    CUDA_VISIBLE_DEVICES=$devices python -m torch.distributed.launch --nproc_per_node=2 \
         train_block_noOrthogonal.py  \
        --lr $lr --batch-size 128 --wd $wd --epochs 5 --datasets $datasets \
        --params_start 0 --params_end 18 --blocks $block \
        --optimizer adamw --print-freq 200 \
        --data-location $DST --model-location ../models --workers 16 \
        --cosine-lr --wd $wd
    done
done