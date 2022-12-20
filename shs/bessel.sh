#!/bin/bash
#SBATCH --job-name=bessel
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 10:10:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=none
#SBATCH -c 128


export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

module load tensorflow
srun python bessel.py