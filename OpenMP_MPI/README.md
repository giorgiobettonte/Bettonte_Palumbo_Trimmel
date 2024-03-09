# Conjugate Gradient with MPI+OpenMP
```bash
$ module load intel 
```

## Compile
```bash
$ mpicxx -O2 -fopenmp -o openmp_mpi openmp_mpi.cpp
```

## Run
Modify the openmp_mpi.cpp to include the desired matrix and rhs, or insert them by command line:
```bash
$ mpiexec -n 4 ./openmp_mpi
