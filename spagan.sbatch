#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00 
#SBATCH --account=panjun
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR
python train.py 
