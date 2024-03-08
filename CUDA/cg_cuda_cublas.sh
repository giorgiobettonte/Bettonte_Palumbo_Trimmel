#!/bin/bash -l
#SBATCH --job-name=cuda_cublas_%j          # name of job
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --time=00:10:00                    # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --output=cuda_cublas_%j.out        # output file
#SBATCH --account=p200301                  # project account
#SBATCH --qos=default                      # SLURM qos

# Load modules
module load intel
module load CUDA/12.2.0

# Compile code
nvcc -O2 cg_cuda_cublas.cu -o cg_cuda_cublas -lcublas

# Run Code
srun -n "${SLURM_NTASKS}" ./cg_cuda_cublas ../io/matrix.bin ../io/rhs.bin ../io/sol.bin

