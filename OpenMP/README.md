# Conjugate Gradient with OpenMP and OpenBLAS
```bash
$ module load intel OpenBLAS
```

## Compile
```bash
$ make conjugate_gradients_openMP && make conjugate_gradients_openMP_BLAS
```

## Run
Modify the Makefile to include the desired matrix and rhs. Then:
1. OpenMP version
```bash
$ make runOpenMP
```

2. OpenMP + OpenBLAS version
```bash
$ make runOpenMP_BLAS
```



