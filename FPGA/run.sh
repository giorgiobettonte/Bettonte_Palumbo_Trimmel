#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --account=p200301
#SBATCH --partition=fpga
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

module load ifpgasdk 520nmx
./vecAdd.exe ../../io/matrix2500.bin ../../io/rhs2500.bin ../../io/sol2500.bin
