#!/bin/bash -l
#SBATCH --job-name=openacc_%j              # name of job
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --time=00:10:00                    # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --output=openacc_%j.out            # output file
#SBATCH --account=p200301                  # project account
#SBATCH --qos=default                      # SLURM qos

# Load modules
module load intel
module load  NVHPC/23.7-CUDA-12.2.0

# Compile code
nvc++ -fast -Minfo=all -acc cg_openacc.cpp -o cg_openacc

# Run Code
srun -n "${SLURM_NTASKS}" ./cg_openacc ../io/matrix.bin ../io/rhs.bin ../io/sol.bin

