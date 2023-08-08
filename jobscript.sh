#!/bin/bash -l

# Request a number of GPUs
#$ -l gpu=1

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=24:00:0

# Request RAM
#$ -l mem=16G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=32G

# Set the name of the job.
#$ -N FedKDD

# Set the working directory
#$ -wd /home/ucabcuf/Scratch/FedKDD

# Set email alerts
#$ -m be

# load modules
module purge
module load default-modules
module unload compilers mpi
module load gcc-libs/4.9.2
module load python/miniconda3/4.10.3

# Activate conda environment
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate FedKDD

# Run nvidia-smi to check the GPU status
nvidia-smi

# Run the program
python3 main.py --dataset-id cinic10 --dataset-path dataset/cinic-10 --data-partition iid --synthetic-path dataset/cinic-10/10K --server-model resnet18 --client-model strategy_1 --epochs 5 --kd-epochs 5 --batch-size 128 --kd-batch-size 128 --num-rounds 100 --num-clients 5 --load-diffusion True --save-checkpoint True
