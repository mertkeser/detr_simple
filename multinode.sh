#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-02:00:00
#SBATCH -o multinode.out
#SBATCH -p develbooster
#SBATCH -A training2203

# activate conda env
# source activate $1

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
module load NCCL/2.12.7-1-CUDA-11.5

# run script from above
srun python ./main.py --batch_size 64 --nodeconfig '2x4' --cp 'multinode-checkpoints/'
