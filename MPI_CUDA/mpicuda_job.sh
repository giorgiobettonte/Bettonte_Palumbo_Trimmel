#!/bin/bash -l
#SBATCH --nodes=2                          # number of nodes
#SBATCH --ntasks=4                      # number of tasks
#SBATCH --ntasks-per-node=2               # number of tasks per node
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --cpus-per-task=1                  # number of cores per task
#SBATCH --time=00:15:00                    # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --account=account                # project account
#SBATCH --qos=default                      # SLURM qos


srun ./mpi_cuda
