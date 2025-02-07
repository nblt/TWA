#!/bin/bash

#SBATCH --mail-user=li.tao@sjtu.edu.cn
#SBATCH --job-name=machine_translation
#SBATCH --partition=a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=log/%j_lt.log
#SBATCH --error=log/z_%j_lt.err

module load miniconda3/4.10.3
source activate syhpc

# export CUDA_LAUNCH_BLOCKING=1

python _adamtrain.py \
	--wd 0.0001 \
	--lr 0.0005 \
	--bs 256 \
	--ts 100000 \
	--dp 0.1 \
	--seed 123

python _adamtrain_twa.py \
	--wd 0.0001 \
	--lr 0.01 \
	--bs 256 \
	--ts 2000 \
	--dp 0.1 \
	--layer-size 1000000000000000 \
	--params_start 1 \
	--params_end 51 \
	--seed 2


