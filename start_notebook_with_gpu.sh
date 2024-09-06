#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --output=jupyter-%J.log
#SBATCH --gres=gpu
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --partition=titans

hostname
source ~/.bashrc
conda activate my_torch_env
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888