#!/bin/bash -l
#SBATCH --job-name=sycl_mkl_%j             # name of job
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --cpus-per-task=1                  # number of gpu per task
#SBATCH --time=00:10:00                    # time (HH:MM:SS)
#SBATCH --partition=cpu                    # partition
#SBATCH --output=sycl_mkl_%j.out           # output file
#SBATCH --account=p200301                  # project account
#SBATCH --qos=default                      # SLURM qos

# Load modules
module load intel
module load intel-compilers
module load imkl/2023.1.0

# Compile code
icpx -fsycl -DMKL_ILP64 -I${MKLROOT}/include cg_sycl_mkl.cpp -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -ltbb -pthread -ldl -lm -o cg_sycl_mkl

# Run Code
srun -n "${SLURM_NTASKS}" ./cg_sycl_mkl ../io/matrix.bin ../io/rhs.bin ../io/sol.bin

