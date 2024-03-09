

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

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
    size_t max_iters = 1000;
    double rel_error = 1e-9;


    //get sources (matrix, rhs, sol)
    const char * input_file_matrix = "../io/matrix.bin"; const char * input_file_rhs = "../io/rhs.bin"; const char * output_file_sol = "../io/sol_openmpi.bin";
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
    double * sol = new double[size];
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
        MPI_File_open(MPI_COMM_WORLD, "../io/sol_openmpi.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
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

    delete[] matrix; delete[] rhs; delete[] sol;

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
    //modify to better balance 
        (rank != mpi_size - 1) ? num_rows = (total_rows / mpi_size) : num_rows = (total_rows / mpi_size) + (total_rows % mpi_size);
    else
        num_rows = total_rows / mpi_size;

    //instantiate a matrix of the correct dimension
    matrix = new double[num_rows * num_cols];

    //move to the correct index of the file (according to own rank)

    if(total_rows % mpi_size != 0)
        fseek(file, (rank * (total_rows / mpi_size) * num_cols) * sizeof(double), SEEK_CUR);
    else
        fseek(file, (rank * num_rows * num_cols) * sizeof(double), SEEK_CUR);
        


    //and read the correct number of doubles (according to own rank)
    fread(matrix, sizeof(double), num_rows * num_cols, file);

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


double dot(const double * x, const double * y, size_t size) {
    double result = 0.0;
#pragma omp parallel for shared(x, y) reduction(+:result) num_threads(16) 
    for(size_t i = 0; i < size; i++) {
        result += x[i] * y[i];
    }

    return result;
}

void axpby(double alpha, const double * x, double beta, double * y, size_t size)
{
#pragma omp parallel for shared(x, y) num_threads(16)
    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols)
{
    for(size_t r = 0; r < num_rows; r++)
    {
        double y_val = 0.0;
#pragma omp parallel for shared(A, x) reduction(+:y_val) num_threads(16) 
        for(size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[c];
        }
        y[r] = beta * y[r] + y_val;
    }
}


bool conjugate_gradients(const double * A, const double * b, double * x, size_t size, size_t total_rows, size_t max_iters, double rel_error)
{   
    //MPI initialization
    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); //get the total number of processes in the communicator


    size_t num_iters;
    double alpha, beta, rr, rr_new; double bb; 
    double *temp1 = new double; double *temp2 = new double; 
    double * p_temp = new double[size]; double * p = new double[total_rows];
    double * Ap_temp = new double[size]; double * Ap = new double[total_rows];
    double * r = new double[size];


#pragma omp paralell for shared(x, r, b, p_temp) num_threads(16)
    for(size_t i = 0; i < size; i++)
    {
        x[i] = 0.0; r[i] = b[i]; p_temp[i] = b[i];
    }

    int * sendcounts = new int[mpi_size];int * displs = new int[mpi_size]; //vectors needed by Allgatherv

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

    

    bb = dot(b, b, size);
    //AllReduce (MPI_SUM) of values bb and store the result in rr
    MPI_Allreduce(&bb, temp1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    bb = *temp1;
    rr = bb;



    //MPI_AllGatherV to compose p from p_temps (store the composition in p)
    MPI_Allgatherv(p_temp, size, MPI_DOUBLE, p, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 1)
        printf("p[0] temp in mpi_cuda rank %i: %f\n", rank, p[0]);
    MPI_Barrier(MPI_COMM_WORLD);


    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {   
        gemv(1.0, A, p, 0.0, Ap_temp, size, total_rows);


        //MPI_AllGatherV to compose Ap from Ap_temps (store the composition in Ap)
        MPI_Allgatherv(Ap_temp, size, MPI_DOUBLE, Ap, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        

        *temp1 = dot(p_temp, Ap_temp, size);
        

        //AllReduce (MPI_SUM) of values temp1 and store the result in temp2
        MPI_Allreduce(temp1, temp2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        

        alpha = rr / *temp2;
        
        axpby(alpha, p_temp, 1.0, x, size);
        axpby(-alpha, Ap_temp, 1.0, r, size);


        
        *temp1 = dot(r, r, size);
        
        //AllReduce (MPI_SUM) of values temp1 and store the result in temp2
        MPI_Allreduce(temp1, temp2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        rr_new = *temp2;
        beta = rr_new / rr;

        rr = rr_new;
        MPI_Barrier(MPI_COMM_WORLD);

        if(std::sqrt(rr / bb) < rel_error) { break; }
        if(rank == 3 && num_iters == 1)
            printf("p_temp[2000]: %f\n", p_temp[2000]);
        axpby(1.0, r, beta, p_temp, size);
        if(rank == 3 && num_iters == 1)
            printf("p_temp[2000]: %f\n", p_temp[2000]);

        //MPI_AllGatherV to compose p from p_temps (store the composition in p)
        MPI_Allgatherv(p_temp, size, MPI_DOUBLE, p, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    if(rank == 0)
    {
        if(num_iters <= max_iters)
            printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        else
            printf("Did NOT converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
    delete[] r; delete[] p; delete[] Ap; delete[] Ap_temp; delete[] p_temp; delete[] temp1; delete[] temp2;
    if(num_iters <= max_iters)
        return true;
    return false;

    
}

