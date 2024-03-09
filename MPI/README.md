# Conjugate Gradient with MPI
```bash
$ module load intel 
```

## Compile
```bash
$ mpicxx -O2 -o mpi mpi.cpp
```

## Run
Modify the Makefile to include the desired matrix and rhs. Then:
```bash
$ mpiexec -n 4 ./cg_mpi
