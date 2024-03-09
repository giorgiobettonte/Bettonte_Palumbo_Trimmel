# Conjugate Gradient with MPI
```bash
$ module load intel 
```

## Compile
```bash
$ mpicxx -O2 -o mpi mpi.cpp
```

## Run
Modify the mpi.cpp to include the desired matrix and rhs, or insert them by command line:
```bash
$ mpiexec -n 4 ./cg_mpi
