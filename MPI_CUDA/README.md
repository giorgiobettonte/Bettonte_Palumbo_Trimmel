  # MPI combined with Cuda.

## Modules to load: 
```bash
$  module load intel
$  module load CUDA/12
```

## Compile
```bash
 $ nvcc -ccbin=mpicxx -O2 -o mpi_cuda src/mpi_cuda.cpp -lcublas
```

## Run:
```bash
 $  sbatch mpicuda_job.sh
```

## To see execution:
```bash
$ cat slurm-<job_number>.out
```
