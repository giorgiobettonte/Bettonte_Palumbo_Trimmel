conjugate_gradients_openMP: conjugate_gradients.cpp
	icpx -O3 conjugate_gradients.cpp -o conjugate_gradients_openMP -fopenmp

conjugate_gradients_openMP_BLAS: conjugate_gradients_BLAS.cpp
	icpx -O3 conjugate_gradients_BLAS.cpp -o conjugate_gradients_openMP_BLAS -fopenmp -I/$EBROOTOPENBLAS/include -L/$EBROOTOPENBLAS/lib -lopenblas -Wall

runOpenMP:
	srun -c 100 ./conjugate_gradients_openMP ../io/matrix10.bin ../io/rhs10.bin ../io/sol10.bin

runOpenMP_BLAS:
	srun -c 100 ./conjugate_gradients_openMP_BLAS ../io/matrix10.bin ../io/rhs10.bin ../io/sol10.bin 1000 1e-9 100
