#!/bin/bash

#SBATCH --job-name=dnn_ensembling
#SBATCH --output=./logs/%J.log
#SBATCH --gres=gpu
#SBATCH --time=3:00:00
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=4
#SBATCH --partition=titans
#SBATCH --exclusive

hostname
source ~/.bashrc
module load CUDA/11.8 CUDNN/8.6
conda activate my_torch_env

# EVAL RELATIVE CURVES

# Process seed 0-1, 1-3, 0-3
# Process curve Bezier, PolyChain
seeds=("0-1")
curves=("PolyChain" "Bezier")
for seed in "${seeds[@]}"; do
  for curve in "${curves[@]}"; do
        #VGG16
        # python3 eval_relative_curve.py --dir=/scratch/fmager/fge --dataset=CIFAR100 --use_test --transform=VGG --data_path=./data --model=VGG16 --curve=$curve --seed_from_to=$seed --num_bends=3 --ckpt=checkpoint-200.pt --sampling_method=linear --num_points=21 --layer_name=fc3 --num_anchors=512 --batch_size=512 --center
        #PreResNet164
        # python3 eval_relative_curve.py --dir=/scratch/fmager/fge --dataset=CIFAR100 --use_test --transform=ResNet --data_path=./data --model=PreResNet164 --curve=$curve --seed_from_to=$seed  --num_bends=3 --ckpt=checkpoint-150.pt --sampling_method=linear --num_points=21 --layer_name=fc --num_anchors=256 --batch_size=64 --center
        #WideResNet28x10
        python3 eval_relative_curve.py --dir=/scratch/fmager/fge --dataset=CIFAR100 --use_test --transform=ResNet --data_path=./data --model=WideResNet28x10 --curve=$curve --seed_from_to=$seed  --num_bends=3 --ckpt=checkpoint-200.pt --sampling_method=linear --num_points=21 --layer_name=linear --num_anchors=640 --batch_size=64 --center
    done
done