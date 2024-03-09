#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <cassert>
#include <chrono>

using namespace std::chrono;

// Function to check CUDA and Cublas errors
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess ) 
    {
        printf("CUDA Runtime Error: %s\n");assert(result == cudaSuccess);
    }
    return result;
}
inline cublasStatus_t checkCublas(cublasStatus_t result) {
    if (result != CUBLAS_STATUS_SUCCESS) 
    {
        printf("CUBLAS Runtime Error: %s\n");assert(result == CUBLAS_STATUS_SUCCESS);
    }
    return result;
}

//functions
bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out, size_t * total_rows_out);
void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file);
void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols);
void axpby(double alpha, const double * x, double beta, double * y, size_t size);
double dot(const double * x, const double * y, size_t size);
bool conjugate_gradients(const double * A, const double * b, double * x, size_t size, size_t total_rows, size_t max_iters, double rel_error);

int main(int argc, char ** argv)
{   
    //MPI initialization
    int rank, mpi_size;
    MPI_Init(nullptr, nullptr);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); //get the total number of processes in the communicator


    //rank 0 prints some initial information
    if(rank == 0){
        printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
        printf("All parameters are optional and have default values\n");
        printf("\n");
    }

    //settings for the conjugate gradient method (explicit setting)
    size_t max_iters = 1000; double rel_error = 1e-9;


    //get sources (matrix, rhs, sol)
    const char * input_file_matrix = "../io/matrix.bin"; const char * input_file_rhs = "../io/rhs.bin"; const char * output_file_sol = "../io/sol_mpicuda.bin";
    if(argc > 1) input_file_matrix = argv[1];if(argc > 2) input_file_rhs = argv[2];if(argc > 3) output_file_sol = argv[3];

    //settings for the conjugate gradient method in case they are passed as arguments by command line
    if(argc > 4) max_iters = atoi(argv[4]);if(argc > 5) rel_error = atof(argv[5]);

    //rank 0 prints some informations regarding sources and settings for the conjugate gradient method
    if(rank == 0){
        printf("Command line arguments:\n");
        printf("  input_file_matrix: %s\n", input_file_matrix);
        printf("  input_file_rhs:    %s\n", input_file_rhs);
        printf("  output_file_sol:   %s\n", output_file_sol);
        printf("  max_iters:         %d\n", max_iters);
        printf("  rel_error:         %e\n", rel_error);
        printf("\n");
    }



    double * matrix; size_t matrix_rows; size_t matrix_cols; 
    double * rhs; size_t rhs_rows; size_t rhs_cols;
    size_t size; size_t total_rows_matrix; size_t total_rows_rhs;

 
    if(rank == 0)printf("Reading matrix from file\n\n");

    bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols, &total_rows_matrix);
    if(!success_read_matrix)printf("Not possible to open the file\n");
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0)printf("Reading right hand side from file\n\n");

    bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols, &total_rows_rhs);
    if(!success_read_rhs)printf("Not possible to open the file\n");
    MPI_Barrier(MPI_COMM_WORLD);

    //controls on matrix and rhs
    if(total_rows_matrix != matrix_cols)return 3;
    if(total_rows_rhs != total_rows_matrix) return 4;
    if(rhs_cols != 1) return 5;

    
    size = matrix_rows; //size will be the number of elements managed by each process

    //find the solution
    // Allocate memory for sol
    double* sol;
    checkCuda(cudaMallocManaged(&sol, size * sizeof(double)));
    double start_time = MPI_Wtime();
    bool convergence = conjugate_gradients(matrix, rhs, sol, size, total_rows_matrix, max_iters, rel_error);
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    if(rank == 0)printf("Time in seconds: %f: ", elapsed_time);


    
    //write the solution in a file
    //open the file
    if(convergence == true)
    {
        MPI_File file;MPI_Status status;
        MPI_File_open(MPI_COMM_WORLD, "../io/sol_mpicuda.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        MPI_File_write(file, &total_rows_rhs, 1, MPI_UNSIGNED, &status); total_rows_rhs = 1;
        MPI_File_write(file, &total_rows_rhs, 1, MPI_UNSIGNED, &status);
        if(total_rows_matrix % mpi_size != 0)
            MPI_File_seek(file, rank * (total_rows_matrix / mpi_size) * sizeof(double) + 2*sizeof(unsigned int), MPI_SEEK_SET);
        else
            MPI_File_seek(file, rank * size * sizeof(double) + 2*sizeof(unsigned int), MPI_SEEK_SET); 

        MPI_File_write(file, sol, size, MPI_DOUBLE, &status);

        // Close file
        MPI_File_close(&file);
    }

    if(rank == 0)printf("Finished successfully\n");

    // Deallocate matrices
    cudaFree(matrix);cudaFree(rhs);cudaFree(sol);
    
    MPI_Finalize();

    return 0;
}


//bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);
bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out, size_t * total_rows_out)
{   
    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); //get the total number of processes in the communicator

    double * matrix;
    size_t total_cols;  
    size_t total_rows; size_t num_rows; size_t num_cols;

    FILE * file = fopen(filename, "rb");if(file == nullptr)return false;


    //each process read the number of total rows and columns
    fread(&total_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);


    //computation to establish how many rows has to have in charge this process
    if(total_rows % mpi_size != 0)

        (rank != mpi_size - 1) ? num_rows = (total_rows / mpi_size) : num_rows = (total_rows / mpi_size) + (total_rows % mpi_size);
    else
        num_rows = total_rows / mpi_size;

    // Allocate memory for matrix on device
    checkCuda(cudaMallocManaged(&matrix, num_rows * num_cols * sizeof(double)));
    

    //move to the correct index of the file (according to own rank)
    if(total_rows % mpi_size != 0)
        fseek(file, (rank * (total_rows / mpi_size) * num_cols) * sizeof(double), SEEK_CUR);
    else
        fseek(file, (rank * num_rows * num_cols) * sizeof(double), SEEK_CUR);

    //and read the correct number of doubles (according to own rank)
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    //de-allocate memory
    *matrix_out = matrix;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;
    *total_rows_out = total_rows;

    fclose(file);

    return true;
}

void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file = stdout)
{
    for(size_t r = 0; r < num_rows; r++)
    {
        for(size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}


bool conjugate_gradients(const double * A, const double * b, double * x, size_t size, size_t total_rows, size_t max_iters, double rel_error)
{   
    // Initialize cuBLAS
    cublasHandle_t handle; checkCublas(cublasCreate(&handle));

    //MPI initialization
    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); //get the total number of processes in the communicator

    //variables ad allocations on the device
    size_t num_iters;
    double alpha, beta, rr, rr_new, bb, result;
    double *temp1; checkCuda(cudaMallocManaged(&temp1, size * sizeof(double)));
    double *temp2; checkCuda(cudaMallocManaged(&temp2, size * sizeof(double)));
    double * r; checkCuda(cudaMallocManaged(&r, size * sizeof(double)));
    double * p_temp; checkCuda(cudaMallocManaged(&p_temp, size * sizeof(double)));
    double * p; checkCuda(cudaMallocManaged(&p, total_rows * sizeof(double)));
    double * Ap_temp; checkCuda(cudaMallocManaged(&Ap_temp, size * sizeof(double)));
    double * Ap; checkCuda(cudaMallocManaged(&Ap, total_rows * sizeof(double)));
    double gemv_1 = 1.0, gemv_2 = 0.0, beta_2 = 1.0;
    
    //initialization of vectors
    checkCuda(cudaMemset(x, 0, size * sizeof(double)));
    checkCuda(cudaMemcpy(r, b, size * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(p_temp, b, size * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());


    int * sendcounts = new int[mpi_size]; int * displs = new int[mpi_size]; //vectors needed by MPI functions
    //initialization of them
    if(total_rows % mpi_size != 0)
        for(int i = 0; i < mpi_size; i++)
            (i != mpi_size - 1) ? sendcounts[i] = (total_rows / mpi_size) : sendcounts[i] = (total_rows / mpi_size) + (total_rows % mpi_size);
    else
        for(int i = 0; i < mpi_size; i++)
            sendcounts[i] = total_rows / mpi_size;


    int sum = 0; displs[0] = 0;
    for (int i = 1; i < mpi_size; i++) 
    {
        displs[i] = sum + sendcounts[i-1];
        sum += sendcounts[i];
    }

    
    //dot productu and Allreduce
    checkCublas(cublasDdot(handle, size, b, 1, b, 1, &bb)); checkCuda(cudaDeviceSynchronize());
    MPI_Allreduce(&bb, temp1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    bb = *temp1; rr = bb;



    //MPI_AllGatherV to compose p from p_temps (store the composition in p)
    MPI_Allgatherv(p_temp, size, MPI_DOUBLE, p, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {   

        checkCublas(cublasDgemv(handle, CUBLAS_OP_T, total_rows, size, &gemv_1, A, total_rows, p, 1, &gemv_2, Ap_temp, 1));
        checkCuda(cudaDeviceSynchronize());
        

        //MPI_AllGatherV to compose Ap from Ap_temps (store the composition in Ap)
        MPI_Allgatherv(Ap_temp, size, MPI_DOUBLE, Ap, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        
        
        // Dot product
        checkCublas(cublasDdot(handle, size, p_temp, 1, Ap_temp, 1, &result)); checkCuda(cudaDeviceSynchronize());
        *temp1 = result;

        //AllReduce (MPI_SUM) of values temp1 and store the result in temp2
        MPI_Allreduce(temp1, temp2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Vector additions
        alpha = rr / *temp2;
        checkCublas(cublasDaxpy(handle, size, &alpha, p_temp, 1, x, 1)); checkCuda(cudaDeviceSynchronize());
        alpha = -1 * alpha;
        checkCublas(cublasDaxpy(handle, size, &alpha, Ap_temp, 1, r, 1)); checkCuda(cudaDeviceSynchronize());
        

        // Dot product
        checkCublas(cublasDdot(handle, size, r, 1, r, 1, &result));checkCuda(cudaDeviceSynchronize());
        *temp1 = result;

        //AllReduce (MPI_SUM) of values temp1 and store the result in temp2
        MPI_Allreduce(temp1, temp2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        //update variables and wait for everybody
        rr_new = *temp2;
        beta = rr_new / rr;
        rr = rr_new;
        MPI_Barrier(MPI_COMM_WORLD);
        
        //check condition on the residual
        if(std::sqrt(rr / bb) < rel_error) { break;}

        // Scaling P by beta
        checkCublas(cublasDscal(handle, size, &beta, p_temp, 1)); checkCuda(cudaDeviceSynchronize());

        //vector addition
        checkCublas(cublasDaxpy(handle, size, &beta_2, r, 1, p_temp, 1)); checkCuda(cudaDeviceSynchronize());
        
        //MPI_AllGatherV to compose p from p_temps (store the composition in p)
        MPI_Allgatherv(p_temp, size, MPI_DOUBLE, p, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // Destroy cuBLAS handle
    checkCublas(cublasDestroy(handle));

    // Free allocated memory
    checkCuda(cudaFree(r));
    checkCuda(cudaFree(p)); checkCuda(cudaFree(p_temp));
    checkCuda(cudaFree(Ap));checkCuda(cudaFree(Ap_temp));
    checkCuda(cudaFree(temp1));checkCuda(cudaFree(temp2));
    
    if(num_iters <= max_iters)
    {
        if(rank == 0)
            printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        return true;
    }
    else
    {
        if(rank == 0)
            printf("NOT Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        return false;
    }
}


