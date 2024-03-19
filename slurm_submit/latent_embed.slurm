#!/bin/bash
#SBATCH --job-name=kevin_job
#SBATCH --output=pretrain_latent_embed.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=12000
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su004-neuralnet

source ~/.bashrc

conda activate react

cd /gpfs/home/x/xk18/PhD_code_exp/project_react_emotion/

srun python pretrain_latent_embed.py --exp_num 2