#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cblas.h>

bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out)
{
    double * matrix;
    size_t num_rows;
    size_t num_cols;

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    matrix = new double[num_rows * num_cols];
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    fclose(file);

    return true;
}



bool write_matrix_to_file(const char * filename, const double * matrix, size_t num_rows, size_t num_cols)
{
    FILE * file = fopen(filename, "wb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fwrite(&num_rows, sizeof(size_t), 1, file);
    fwrite(&num_cols, sizeof(size_t), 1, file);
    fwrite(matrix, sizeof(double), num_rows * num_cols, file);

    fclose(file);

    return true;
}



void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file = stdout)
{
    fprintf(file, "%zu %zu\n", num_rows, num_cols);
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


void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error, int num_partitions)
{
    double alpha, beta, bb, rr, rr_new;
    double * r = new double[size]; 
    double * p = new double[size];
    double * Ap = new double[size];
    int num_iters;

    #pragma omp parallel for
    for(size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    bb = cblas_ddot(size, b, 1, b, 1);
    rr = bb;

    // 1D partitioning
    size_t part_size = size / num_partitions;

    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        // Ap
        #pragma omp parallel for
        for(size_t i = 0; i < num_partitions; i++) {
            if(i == num_partitions - 1 && size % num_partitions != 0)
                cblas_dgemv(CblasRowMajor, CblasNoTrans, part_size + size % num_partitions, size, 1.0, &A[i * part_size * size], size, p, 1, 0.0, &Ap[i * part_size], 1);
            else
                cblas_dgemv(CblasRowMajor, CblasNoTrans, part_size, size, 1.0, &A[i * part_size * size], size, p, 1, 0.0, &Ap[i * part_size], 1);
        }

        // p [dot] Ap
        double dotSum = 0.0;
        #pragma omp parallel for reduction(+:dotSum)
        for(size_t i = 0; i < num_partitions; i++) {
            if(i == num_partitions - 1 && size % num_partitions != 0)
                dotSum += cblas_ddot(part_size + size % num_partitions, &p[i * part_size], 1, &Ap[i * part_size], 1);
            else
                dotSum += cblas_ddot(part_size, &p[i * part_size], 1, &Ap[i * part_size], 1);
        }
        alpha = rr / dotSum;

        // First axpby
        #pragma omp parallel for
        for(size_t i = 0; i < num_partitions; i++) {
            if(i == num_partitions - 1 && size % num_partitions != 0)
                cblas_daxpby(part_size + size % num_partitions, alpha, &p[i * part_size], 1, 1.0, &x[i * part_size], 1);
            else
                cblas_daxpby(part_size, alpha, &p[i * part_size], 1, 1.0, &x[i * part_size], 1);
        }
        
        // Second axpby
        #pragma omp parallel for
        for(size_t i = 0; i < num_partitions; i++) {
            if(i == num_partitions - 1 && size % num_partitions != 0)
                cblas_daxpby(part_size + size % num_partitions, -alpha, &Ap[i * part_size], 1, 1.0, &r[i * part_size], 1);
            else
                cblas_daxpby(part_size, -alpha, &Ap[i * part_size], 1, 1.0, &r[i * part_size], 1);
        }

        // r [dot] r
        rr_new = 0.0;
        #pragma omp parallel for reduction(+:rr_new)
        for(size_t i = 0; i < num_partitions; i++) {
            if(i == num_partitions - 1 && size % num_partitions != 0)
                rr_new += cblas_ddot(part_size + size % num_partitions, &r[i * part_size], 1, &r[i * part_size], 1);
            else
                rr_new += cblas_ddot(part_size, &r[i * part_size], 1, &r[i * part_size], 1);
        }

        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / bb) < rel_error) { break; }

        // Last axpby
        #pragma omp parallel for
        for(size_t i = 0; i < num_partitions; i++) {
            if(i == num_partitions - 1 && size % num_partitions != 0)
                cblas_daxpby(part_size + size % num_partitions, 1.0, &r[i * part_size], 1, beta, &p[i * part_size], 1);
            else
                cblas_daxpby(part_size, 1.0, &r[i * part_size], 1, beta, &p[i * part_size], 1);
        }
    }

    delete[] r;
    delete[] p;
    delete[] Ap;

    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
}





int main(int argc, char ** argv)
{
    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");

    const char * input_file_matrix = "io/matrix.bin";
    const char * input_file_rhs = "io/rhs.bin";
    const char * output_file_sol = "io/sol.bin";
    int max_iters = 1000;
    double rel_error = 1e-9;
    int num_partitions = 4;

    if(argc > 1) input_file_matrix = argv[1];
    if(argc > 2) input_file_rhs = argv[2];
    if(argc > 3) output_file_sol = argv[3];
    if(argc > 4) max_iters = atoi(argv[4]);
    if(argc > 5) rel_error = atof(argv[5]);
    if(argc > 6) num_partitions = atoi(argv[6]);

    printf("Command line arguments:\n");
    printf("  input_file_matrix: %s\n", input_file_matrix);
    printf("  input_file_rhs:    %s\n", input_file_rhs);
    printf("  output_file_sol:   %s\n", output_file_sol);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("  num_partitions:    %d\n", num_partitions);
    printf("\n");



    double * matrix;
    double * rhs;
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
        if(rhs_cols != 1)
        {
            fprintf(stderr, "Right hand side has to have just a single column\n");
            return 5;
        }
        if (num_partitions > matrix_rows) 
        {
            fprintf(stderr, "Too many partitions\n");
            return 6;
        }
        size = matrix_rows;
    }

    int num_threads = omp_get_max_threads();
    printf("Using %d OpenMP threads\n", num_threads);
    printf("Solving the system ...\n");
    double * sol = new double[size];

    // Starting chrono
    const auto start{std::chrono::steady_clock::now()};
    conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error, num_partitions);
    // Ending chrono
    const auto end{std::chrono::steady_clock::now()};

    // Total time
    const std::chrono::duration<double> elapsed_seconds{end - start};

    
    printf("Done\n");
    printf("\n");

    printf("Writing solution to file ...\n");
    bool success_write_sol = write_matrix_to_file(output_file_sol, sol, size, 1);
    if(!success_write_sol)
    {
        fprintf(stderr, "Failed to save solution\n");
        return 6;
    }
    printf("Done\n"); 
    printf("\n");

    delete[] matrix;
    delete[] rhs;
    delete[] sol;

    printf("Finished successfully\n");

    printf("Elapsed time: %e\n", elapsed_seconds.count());

    return 0;
}
