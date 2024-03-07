#!/bin/bash -l
#SBATCH --nodes=1                         # number of nodes
#SBATCH --ntasks=4                        # number of tasks
#SBATCH --qos=default                     # SLURM qos
#SBATCH --ntasks-per-node=16               # number of tasks per node
#SBATCH --cpus-per-task=1                 # number of cores per task
#SBATCH --time=00:15:00                   # time (HH:MM:SS)
#SBATCH --partition=cpu                   # partition
#SBATCH --account=account                 # project account

srun ./openmp_mpi
