#!/bin/bash

#SBATCH --job-name=dnn_ensembling
#SBATCH --output=./logs/%J.log
#SBATCH --gres=gpu
#SBATCH --time=8:00:00
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=4
#SBATCH --partition=titans
#SBATCH --exclusive

hostname
source ~/.bashrc
module load CUDA/11.8 CUDNN/8.6
conda activate my_torch_env


# TRAIN ENDPOINTS

#VGG16
# python3 train.py --dir=/scratch/fmager/fge --dataset=CIFAR100 --data_path=./data --model=VGG16 --epochs=200 --lr=0.05 --wd=5e-4 --use_test --transform=VGG --seed=3
#PreResNet164
# python3 train.py --dir=/scratch/fmager/fge --dataset=CIFAR100 --data_path=./data  --model=PreResNet164 --epochs=150  --lr=0.1 --wd=3e-4 --use_test --transform=ResNet --seed=3
#WideResNet28x10
# python3 train.py --dir=/scratch/fmager/fge --dataset=CIFAR100 --data_path=./data --model=WideResNet28x10 --epochs=200 --lr=0.1 --wd=5e-4 --use_test --transform=ResNet --seed=3