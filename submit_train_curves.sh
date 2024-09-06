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

# TRAIN CURVES

#VGG16
# python3 train.py --dir=/scratch/fmager/fge --dataset=CIFAR100 --use_test --transform=VGG --data_path=./data --model=VGG16 --curve=PolyChain --num_bends=3  --init_start=seed_1/checkpoint-200.pt --init_end=seed_3/checkpoint-200.pt --fix_start --fix_end --epochs=200 --lr=0.015 --wd=5e-4
#PreResNet164
# python3 train.py --dir=/scratch/fmager/fge --dataset=CIFAR100 --use_test --transform=ResNet --data_path=./data --model=PreResNet164 --curve=PolyChain --num_bends=3  --init_start=seed_0/checkpoint-150.pt --init_end=seed_3/checkpoint-150.pt --fix_start --fix_end --epochs=150 --lr=0.03 --wd=3e-4
#WideResNet28x10
# python3 train.py --dir=/scratch/fmager/fge --dataset=CIFAR100 --use_test --transform=ResNet --data_path=./data --model=WideResNet28x10 --curve=Bezier --num_bends=3  --init_start=seed_0/checkpoint-200.pt --init_end=seed_1/checkpoint-200.pt --fix_start --fix_end --epochs=200 --lr=0.03 --wd=5e-4
