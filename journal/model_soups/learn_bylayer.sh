#!/bin/bash

#SBATCH --mail-user=li.tao@sjtu.edu.cn
#SBATCH --job-name=twa_soup
#SBATCH --partition=a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --output=log/%j_lt.log
#SBATCH --error=log/z_%j_lt.err

module load miniconda3/4.10.3
source activate syhpc

for lr in 0.05
do
CUDA_VISIBLE_DEVICES=0 python learned_bylayer.py --batch-size 256 --lr $lr \
    --data-location /dssg/home/acct-eehxl/eehxl-yzx/data/ --model-location ../models
CUDA_VISIBLE_DEVICES=0 python learned_bylayer.py --batch-size 256 --lr $lr --cosine-lr \
    --data-location /dssg/home/acct-eehxl/eehxl-yzx/data/ --model-location ../models
done