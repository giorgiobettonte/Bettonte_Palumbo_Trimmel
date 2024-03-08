#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>


// Define a structure to hold scalar variables
struct SolverData {
    double *a;
    double *bb;
    double *rr;
    double *rr_new;
    double *r;
    double *p;
    double *Ap;
};

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

// Function to allocate memory on device for solver data
void allocateSolverData(SolverData *&solverData, size_t size) {
    solverData = new SolverData;

    // Allocate memory for scalar variables
    cudaMallocManaged(&solverData->a, sizeof(double));
    cudaMallocManaged(&solverData->bb, sizeof(double));
    cudaMallocManaged(&solverData->rr, sizeof(double));
    cudaMallocManaged(&solverData->rr_new, sizeof(double));

    // Allocate memory for vectors r, p, and Ap
    cudaMallocManaged(&solverData->r, size * sizeof(double));
    cudaMallocManaged(&solverData->p, size * sizeof(double));
    cudaMallocManaged(&solverData->Ap, size * sizeof(double));
}

// Function to free memory allocated for solver data
void freeSolverData(SolverData *&solverData) {
    cudaFree(solverData->a);
    cudaFree(solverData->bb);
    cudaFree(solverData->rr);
    cudaFree(solverData->rr_new);
    cudaFree(solverData->r);
    cudaFree(solverData->p);
    cudaFree(solverData->Ap);

    delete solverData;
    solverData = nullptr;
}

// Function to check CUDA errors
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }

    return result;
}

// CUDA atomic addition for double precision
__device__
double doubleAtomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// CUDA function to initialize
__global__
void initializeArrays(double* x, const double* b, double* r, double* p, size_t size) {
    // Set index and stride according GPU block and thread
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < size; i += stride) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }
}

// CUDA function to perform dot product
__global__
void dot(const double* x, const double* y, double* result, size_t size) {
    // Set index and stride according GPU block and thread
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    double partialSum = 0.0;

    for (size_t i = index; i < size; i += stride)
        partialSum += x[i] * y[i];

    doubleAtomicAdd(result, partialSum);
}

// CUDA function to perform axpby
__global__
void axpby(double alpha, const double* x, double beta, double* y, size_t size) {
    // Set index and stride according GPU block and thread
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < size; i += stride)
        y[i] = alpha * x[i] + beta * y[i];
}

// CUDA function to perform gemv
__global__
void gemv(const double* A, const double* x, double* y, size_t num_rows, size_t num_cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        double y_val = 0.0;

        for (size_t c = 0; c < num_cols; c++)
            y_val += A[row * num_cols + c] * x[c];

        y[row] = y_val;
    }
}

// Function to perform conjugate gradient on input matrix
void conjugate_gradients(const double* A, const double* b, double* x, size_t size, int max_iters, double rel_error) {
    // Set threads and block size according to matrix size
    const size_t N = size;
    size_t threadsPerBlock = 256;
    size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Define variables
    int num_iters;
    double alpha, beta;
    SolverData *solverData;

    // Allocate memory for solver data
    allocateSolverData(solverData, size);

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Initialize arrays asynchronously using stream1
    initializeArrays<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(x, b, solverData->r, solverData->p, size);

    // Compute dot product of b and b asynchronously using stream2
    dot<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(b, b, solverData->bb, size);

    // Synchronize stream2
    cudaStreamSynchronize(stream2);

    *solverData->rr = *solverData->bb;

    // Synchronize stream1
    cudaStreamSynchronize(stream1);

    for (num_iters = 1; num_iters <= max_iters; num_iters++) {
        // Compute gemv asynchronously using stream2
        gemv<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(A, solverData->p, solverData->Ap, size, size);

        // Compute dot product asynchronously using stream2
        dot<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(solverData->p, solverData->Ap, solverData->a, size);

        // Synchronize to ensure completion of gemv and dot product
        cudaStreamSynchronize(stream2);

        // Calculate new alpha
        alpha = *solverData->rr / *solverData->a;
        *solverData->a = 0;

        // Compute axpby asynchronously using stream1 and stream2
        axpby<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(alpha, solverData->p, 1.0, x, size);
        axpby<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(-alpha, solverData->Ap, 1.0, solverData->r, size);

        // Compute dot product asynchronously using stream2
        dot<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(solverData->r, solverData->r, solverData->rr_new, size);

        // Synchronize to ensure completion of axpby and dot product
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        // Calculate new beta
        beta = *solverData->rr_new / *solverData->rr;
        *solverData->rr = *solverData->rr_new;
        *solverData->rr_new = 0;

        // Calculate relative error
        if (std::sqrt(*solverData->rr / *solverData->bb) < rel_error) break;

        // Compute axpby asynchronously using stream2
        axpby<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(1.0, solverData->r, beta, solverData->p, size);
    }

    if(num_iters <= max_iters)
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(*solverData->rr / *solverData->bb));
    else
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(*solverData->rr / *solverData->bb));

    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    // Deallocate solver
    freeSolverData(solverData);
}

int main(int argc, char ** argv) {
    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");

    const char* input_file_matrix = "io/matrix.bin";
    const char* input_file_rhs = "io/rhs.bin";
    const char* output_file_sol = "io/sol.bin";
    int max_iters = 1000;
    double rel_error = 1e-9;

    if(argc > 1) input_file_matrix = argv[1];
    if(argc > 2) input_file_rhs = argv[2];
    if(argc > 3) output_file_sol = argv[3];
    if(argc > 4) max_iters = atoi(argv[4]);
    if(argc > 5) rel_error = atof(argv[5]);

    printf("Command line arguments:\n");
    printf("  input_file_matrix: %s\n", input_file_matrix);
    printf("  input_file_rhs:    %s\n", input_file_rhs);
    printf("  output_file_sol:   %s\n", output_file_sol);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("\n");

    // Define variables
    double* matrix;
    double* rhs;
    size_t size;

    {
        printf("Reading matrix from file ...\n");

        size_t matrix_rows;
        size_t matrix_cols;

        bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);

        if(!success_read_matrix)
        {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }
        printf("Done\n");
        printf("\n");


        printf("Reading right hand side from file ...\n");

        size_t rhs_rows;
        size_t rhs_cols;

        bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);

        if(!success_read_rhs)
        {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }

        printf("Done\n");
        printf("\n");


        if(matrix_rows != matrix_cols)
        {
            fprintf(stderr, "Matrix has to be square\n");
            return 3;
        }
        if(rhs_rows != matrix_rows)
        {
            fprintf(stderr, "Size of right hand side does not match the matrix\n");
            return 4;
        }
        if(rhs_cols != 1) {
            fprintf(stderr, "Right hand side has to have just a single column\n");
            return 5;
        }

        // Set size of matrix
        size = matrix_rows;
    }

    // Allocate memory for sol on device
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
