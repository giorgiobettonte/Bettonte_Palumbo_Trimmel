#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

// Function to check CUDA errors
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// Function to read matrix from file
bool read_matrix_from_file(const char* filename, double** matrix_out, size_t* num_rows_out, size_t* num_cols_out) {
    double* matrix;
    size_t num_rows;
    size_t num_cols;

    FILE* file = fopen(filename, "rb");
    if (file == nullptr) {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);

    // Allocate memory for matrix on device
    cudaMallocManaged(&matrix, num_rows * num_cols * sizeof(double));

    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    fclose(file);

    return true;
}

// Function to write matrix to file
bool write_matrix_to_file(const char* filename, const double* matrix, size_t num_rows, size_t num_cols) {
    FILE* file = fopen(filename, "wb");
    if (file == nullptr) {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fwrite(&num_rows, sizeof(size_t), 1, file);
    fwrite(&num_cols, sizeof(size_t), 1, file);
    fwrite(matrix, sizeof(double), num_rows * num_cols, file);

    fclose(file);

    return true;
}

// Function to print matrix
void print_matrix(const double* matrix, size_t num_rows, size_t num_cols, FILE* file = stdout) {
    fprintf(file, "%zu %zu\n", num_rows, num_cols);

    for (size_t r = 0; r < num_rows; r++) {
        for (size_t c = 0; c < num_cols; c++) {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}

// Function to perform conjugate gradient on input matrix
void conjugate_gradients(const double* A, const double* b, double* x, size_t size, int max_iters, double rel_error) {
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Define variables
    int num_iters;
    double alpha, alpha_2, beta, bb, rr, rr_new, rel_err_new;
    double gemv_1 = 1.0, gemv_2 = 0.0, beta_2 = 1.0;
    double *r, *p, *Ap;

    // Allocate memory for vectors r, p, and Ap on device
    cudaMallocManaged(&r, size * sizeof(double));
    cudaMallocManaged(&p, size * sizeof(double));
    cudaMallocManaged(&Ap, size * sizeof(double));

    // Initialize arrays
    /* for (size_t i = 0; i < size; ++i) {
         x[i] = 0.0;
         r[i] = b[i];
         p[i] = b[i];
     }*/

    cudaMemset(x, 0, size * sizeof(double));
    cudaMemcpy(r, b, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(p, b, size * sizeof(double), cudaMemcpyHostToDevice);


    // Dot product of b and b
    cublasDdot(handle, size, b, 1, b, 1, &bb);

    // Copy bb to rr
    rr = bb;

    for (num_iters = 1; num_iters <= max_iters; num_iters++) {
        // Matrix-vector multiplication
        cublasDgemv(handle, CUBLAS_OP_N, size, size, &gemv_1, A, size, p, 1, &gemv_2, Ap, 1);

        // Dot product
        cublasDdot(handle, size, p, 1, Ap, 1, &alpha);

        // Calculate new alpha
        alpha = rr / alpha;
        alpha_2 = -alpha;

        // Vector addition
        cublasDaxpy(handle, size, &alpha, p, 1, x, 1);
        cublasDaxpy(handle, size, &alpha_2, Ap, 1, r, 1);
        cudaDeviceSynchronize();

        // Dot product
        cublasDdot(handle, size, r, 1, r, 1, &rr_new);

        // Calculate new beta
        beta = rr_new / rr;
        rr = rr_new;

        rel_err_new = std::sqrt(rr / bb);
        if (rel_err_new < rel_error) break;

        // Scaling P by beta
        cublasDscal(handle, size, &beta, p, 1);

        // Adding R to the scaled P
        cublasDaxpy(handle, size, &beta_2, r, 1, p, 1);
    }

    if (num_iters <= max_iters)
        printf("Converged in %d iterations, relative error is %e\n", num_iters, rel_err_new);
    else
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, rel_err_new);

    // Destroy cuBLAS handle
    cublasDestroy(handle);

    // Free allocated memory
    cudaFree(r);
    cudaFree(p);
    cudaFree(Ap);
}

int main(int argc, char** argv) {
    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n\n");

    const char* input_file_matrix = "io/matrix.bin";
    const char* input_file_rhs = "io/rhs.bin";
    const char* output_file_sol = "io/sol.bin";
    int max_iters = 1000;
    double rel_error = 1e-9;

    if (argc > 1) input_file_matrix = argv[1];
    if (argc > 2) input_file_rhs = argv[2];
    if (argc > 3) output_file_sol = argv[3];
    if (argc > 4) max_iters = atoi(argv[4]);
    if (argc > 5) rel_error = atof(argv[5]);

    printf("Command line arguments:\n");
    printf("  input_file_matrix: %s\n", input_file_matrix);
    printf("  input_file_rhs:    %s\n", input_file_rhs);
    printf("  output_file_sol:   %s\n", output_file_sol);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n\n", rel_error);

    // Define variables
    double* matrix;
    double* rhs;
    size_t size;

    {
        printf("Reading matrix from file ...\n");

        size_t matrix_rows;
        size_t matrix_cols;

        bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);

        if (!success_read_matrix) {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }
        printf("Done\n\n");

        printf("Reading right hand side from file ...\n");

        size_t rhs_rows;
        size_t rhs_cols;

        bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);

        if (!success_read_rhs) {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }

        printf("Done\n\n");

        if (matrix_rows != matrix_cols) {
            fprintf(stderr, "Matrix has to be square\n");
            return 3;
        }
        if (rhs_rows != matrix_rows) {
            fprintf(stderr, "Size of right hand side does not match the matrix\n");
            return 4;
        }
        if (rhs_cols != 1) {
            fprintf(stderr, "Right hand side has to have just a single column\n");
            return 5;
        }

        // Get size of matrix
        size = matrix_rows;
    }

    // Allocate memory for sol
    double* sol;
    cudaMallocManaged(&sol, size * sizeof(double));

    printf("Solving the system ...\n");

    conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error);

    printf("Writing solution to file ...\n");

    bool success_write_sol = write_matrix_to_file(output_file_sol, sol, size, 1);

    if (!success_write_sol) {
        fprintf(stderr, "Failed to save solution\n");
        return 6;
    }

    // Deallocate matrices
    cudaFree(matrix);
    cudaFree(rhs);
    cudaFree(sol);

    printf("Finished successfully\n");

    return 0;
}
