#!/bin/bash

#SBATCH --mail-user=li.tao@sjtu.edu.cn
#SBATCH --job-name=twa_soup
#SBATCH --partition=a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=log/%j_lt.log
#SBATCH --error=log/z_%j_lt.err

module load miniconda3/4.10.3
source activate syhpc


CUDA_VISIBLE_DEVICES=0 python main.py --greedy-soup --data-location /home/dev/ --model-location ../models