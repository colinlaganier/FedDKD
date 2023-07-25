#!/bin/bash -l

# Batch script to run a GPU job under SGE.

# Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=1

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=20:00:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=16G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=15G

# Set the name of the job.
#$ -N FedKDD

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucabcuf/Scratch/FedKDD

# Change into temporary directory to run work
cd $TMPDIR

# load the cuda module (in case you are running a CUDA program)
module purge
module load defaut-modules
module unload compilers mpi
module load python/miniconda3/4.10.3

# Activate conda environment
source $UCL_CONDA_PATH/etc/profile.d/conda.sh 
conda activate FedKDD

# Run the application - the line below is just a random example.
nvidia-smi
python3 python main.py --dataset-id cinic10 --dataset-path dataset/cinic-10 --data-partition iid 
--synthetic-path dataset/cinic-10/synthetic --server-model resnet18 --client-model strategy_1 --epochs 10 
--kd-epochs 10 --batch-size 128 --kd-batch-size 128 --num-rounds 100 --num-clients 5 --load-diffusion True 
--save-checkpoint True

# 10. Preferably, tar-up (archive) all output files onto the shared scratch area
# tar zcvf $HOME/Scratch/files_from_job_$JOB_ID.tar.gz $TMPDIR

# Make sure you have given enough time for the copy to complete!
