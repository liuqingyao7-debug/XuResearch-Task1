#!/bin/bash
#SBATCH --job-name=busi_resnet50
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out

cd ~/projects/research_training
/mnt/cv_data/users/seven/envs/ml/bin/python -u train.py