#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cstring>
#include "CL/cl.h"
#include "AOCLUtils/aocl_utils.h"
#include <chrono>


#define AOCL_ALIGNMENT 64
#define DEBUG 0

using namespace aocl_utils;

static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_program program = NULL;
static cl_kernel dotKernel = NULL;
static cl_kernel axpbyKernel = NULL;
static cl_kernel gemvKernel = NULL;
static cl_int status = 0;
static cl_uint numPlatforms = 0;

cl_mem x_in, y_in;

double dot(const double * a, const double * b, int dim, int numElem);
void axpby(double alpha, double * x, double beta, double * y, int dim);
void gemv(double alpha, const double * A, double * x, double beta, double * y, int num_rows, int num_cols);
bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out);
bool write_matrix_to_file(const char * filename, const double * matrix, size_t num_rows, size_t num_cols);
void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file);
void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error);
void checkErr(cl_int, std::string, std::string success);
void cleanup();

int main( int argc, char* argv[] )
{
    status = clGetPlatformIDs( 1, &platform, &numPlatforms );

    if (status == CL_SUCCESS)
        printf("%u platform(s) found\n", numPlatforms);
    else{
        printf("clGetPlatformIDs(%i)\n", status);
        return 1;
    }
    
    // Get device id
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, NULL);
    checkErr(status, "Failed to get device", "Device found");

    // Create the context.
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkErr(status, "Failed to create context", "Context created");
    
    // Create the command queue.
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkErr(status, "Failed to create command queue", "Command queue created");

    // Create the program.
    std::string binary_file = getBoardBinaryFile("device/CG", device);
    printf("Using AOCX: %s\n\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkErr(status, "Failed to build program", "Program created");

    // Create the kernels - name passed in here must match kernel name in the
    // original CL file, that was compiled into an AOCX file using the AOC tool
    dotKernel = clCreateKernel(program, "dot", &status);
    checkErr(status, "Failed to create kernel", "Kernel created");

    axpbyKernel = clCreateKernel(program, "axpby", &status);
    checkErr(status, "Failed to create kernel", "Kernel created");

    gemvKernel = clCreateKernel(program, "gemv", &status);
    checkErr(status, "Failed to create kernel", "Kernel created");

    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");

    const char * input_file_matrix = "io/matrix.bin";
    const char * input_file_rhs = "io/rhs.bin";
    const char * output_file_sol = "io/sol.bin";
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



    double * matrix;
    double * rhs;

    void * alignedMatrix = NULL;
    void * alignedRhs = NULL;
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

        posix_memalign (&alignedMatrix, AOCL_ALIGNMENT, matrix_cols * matrix_rows * sizeof(double));
        alignedMatrix = matrix;
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
        posix_memalign (&alignedRhs, AOCL_ALIGNMENT, rhs_cols * rhs_cols * sizeof(double));
        alignedRhs = rhs;
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

        size = matrix_rows;
    }

    printf("Solving the system ...\n");
    double * sol = new double[size];

    // Starting chrono
    const auto start{std::chrono::steady_clock::now()};
    conjugate_gradients((double *)alignedMatrix, (double *)alignedRhs, sol, size, max_iters, rel_error);
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

    clReleaseKernel(dotKernel); 
    clReleaseKernel(axpbyKernel);
    clReleaseKernel(gemvKernel);
    clReleaseCommandQueue(queue);

    return 0;
}

double dot(const double * x, const double * y, int dim, int numElem = 100) 
{
    double * c = new double[dim / numElem];
    cl_mem c_out;

    // Create device buffers - assign the buffers in different banks for more efficient
    // memory access 
    // Input
    x_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * dim, NULL, &status);
    checkErr(status, "Failed to allocate input device buffer for x", "Input device buffer for x allocated");
    y_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * dim, NULL, &status);
    checkErr(status, "Failed to allocate input device buffer for y", "Input device buffer for y allocated");
    // Output
    c_out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * dim / numElem, NULL, &status);
    checkErr(status, "Failed to allocate outpud device buffer for c", "Output device buffer for c allocated");

    // Copy data from host to device
    status = clEnqueueWriteBuffer(queue, x_in, CL_TRUE, 0, sizeof(double) * dim, x, 0, NULL, NULL);
    checkError(status, "Failed to copy data x to device", "Copied data x to device");
    status = clEnqueueWriteBuffer(queue, y_in, CL_TRUE, 0, sizeof(double) * dim, y, 0, NULL, NULL);
    checkError(status, "Failed to copy data y to device", "Copied data y to device");

    // Set kernel args
    status = clSetKernelArg(dotKernel, 0, sizeof(cl_mem), (void *)&x_in);
    checkErr(status, "Failed to set arg a", "Arg x setted");
    status = clSetKernelArg(dotKernel, 1, sizeof(cl_mem), (void *)&y_in);
    checkErr(status, "Failed to set arg b", "Arg y setted");
    status = clSetKernelArg(dotKernel, 2, sizeof(cl_mem), (void *)&c_out);
    checkErr(status, "Failed to set arg c", "Arg c setted");
    status = clSetKernelArg(dotKernel, 3, sizeof(unsigned int), (void *)&dim);
    checkErr(status, "Failed to set arg dim", "Arg dim setted");
    status = clSetKernelArg(dotKernel, 4, sizeof(unsigned int), (void *)&numElem);
    checkErr(status, "Failed to set arg numElem", "Arg numElem setted");

    // Launch the kernel - we launch a single work item hence enqueue a task
    //status = clEnqueueTask(queue, kernel, 0, NULL, NULL);
    size_t global_work_size = dim / numElem;
    status = clEnqueueNDRangeKernel(queue, dotKernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    checkErr(status, "Failed to launch kernel", "Kernel launched");

    // Wait for command queue to complete pending events
    status = clFinish(queue);
    checkErr(status, "Failed to finish queue", "Queue finished");

    // Copy results from device to host
    status = clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(double) * dim / numElem, c, 0, NULL, NULL);
    checkErr(status, "Failed to copy data from device", "Data copied from device");

    double toReturn = 0.0;
    
    for(size_t i = 0; i < dim / numElem; i++){
        toReturn += c[i];
    }

    status = clReleaseMemObject(x_in);
    checkErr(status, "Failed to release x", "x released");
    status = clReleaseMemObject(y_in);
    checkErr(status, "Failed to release y", "y released");
    status = clReleaseMemObject(c_out);
    checkErr(status, "Failed to release c", "c released");

    return toReturn;
}

void axpby(double alpha, double * x, double beta, double * y, int dim)
{
    // Create device buffers - assign the buffers in different banks for more efficient
    // memory access 
    // Input
    x_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * dim, NULL, &status);
    checkErr(status, "Failed to allocate input device buffer for x", "Input device buffer for x allocated");
    y_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * dim, NULL, &status);
    checkErr(status, "Failed to allocate input device buffer for y", "Input device buffer for y allocated");

    // Copy data from host to device
    status = clEnqueueWriteBuffer(queue, x_in, CL_TRUE, 0, sizeof(double) * dim, x, 0, NULL, NULL);
    checkError(status, "Failed to copy data x to device", "Copied data x to device");
    status = clEnqueueWriteBuffer(queue, y_in, CL_TRUE, 0, sizeof(double) * dim, y, 0, NULL, NULL);
    checkError(status, "Failed to copy data y to device", "Copied data y to device");

    // Set kernel args
    status = clSetKernelArg(axpbyKernel, 0, sizeof(double), (void *)&alpha);
    checkErr(status, "Failed to set arg alpha", "Arg alpha setted");
    status = clSetKernelArg(axpbyKernel, 1, sizeof(cl_mem), (void *)&x_in);
    checkErr(status, "Failed to set arg x", "Arg x setted");
    status = clSetKernelArg(axpbyKernel, 2, sizeof(double), (void *)&beta);
    checkErr(status, "Failed to set arg beta", "Arg beta setted");
    status = clSetKernelArg(axpbyKernel, 3, sizeof(cl_mem), (void *)&y_in);
    checkErr(status, "Failed to set arg y", "Arg y setted");
    status = clSetKernelArg(axpbyKernel, 4, sizeof(int), (void *)&dim);
    checkErr(status, "Failed to set arg dim", "Arg dim setted");

    // Launch the kernel - we launch a single work item hence enqueue a task
    //status = clEnqueueTask(queue, kernel, 0, NULL, NULL);
    size_t global_work_size = dim;
    status = clEnqueueNDRangeKernel(queue, axpbyKernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    checkErr(status, "Failed to launch kernel", "Kernel launched");

    // Wait for command queue to complete pending events
    status = clFinish(queue);
    checkErr(status, "Failed to finish queue", "Queue finished");

    // Copy results from device to host
    status = clEnqueueReadBuffer(queue, y_in, CL_TRUE, 0, sizeof(double) * dim, y, 0, NULL, NULL);
    checkErr(status, "Failed to copy data from device", "Data copied from device");

    status = clReleaseMemObject(x_in);
    checkErr(status, "Failed to release x", "x released");
    status = clReleaseMemObject(y_in);
    checkErr(status, "Failed to release y", "y released");
}

void gemv(double alpha, const double * A, double * x, double beta, double * y, int num_rows, int num_cols)
{
    cl_mem A_in;

    // y = alpha * A * x + beta * y;
    // Create device buffers - assign the buffers in different banks for more efficient
    // memory access 
    // Input
    A_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * num_cols * num_rows, NULL, &status);
    checkErr(status, "Failed to allocate input device buffer for A", "Input device buffer for A allocated");
    x_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * num_cols, NULL, &status);
    checkErr(status, "Failed to allocate input device buffer for x", "Input device buffer for x allocated");
    y_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * num_cols, NULL, &status);
    checkErr(status, "Failed to allocate input device buffer for y", "Input device buffer for y allocated");

    // Copy data from host to device
    status = clEnqueueWriteBuffer(queue, A_in, CL_TRUE, 0, sizeof(double) * num_cols * num_rows, A, 0, NULL, NULL);
    checkError(status, "Failed to copy data A to device", "Copied data A to device");
    status = clEnqueueWriteBuffer(queue, x_in, CL_TRUE, 0, sizeof(double) * num_cols, x, 0, NULL, NULL);
    checkError(status, "Failed to copy data x to device", "Copied data x to device");
    status = clEnqueueWriteBuffer(queue, y_in, CL_TRUE, 0, sizeof(double) * num_cols, y, 0, NULL, NULL);
    checkError(status, "Failed to copy data y to device", "Copied data y to device");

    // Set kernel args
    status = clSetKernelArg(gemvKernel, 0, sizeof(double), (void *)&alpha);
    checkErr(status, "Failed to set arg alpha", "Arg alpha setted");
    status = clSetKernelArg(gemvKernel, 1, sizeof(cl_mem), (void *)&A_in);
    checkErr(status, "Failed to set arg A", "Arg A setted");
    status = clSetKernelArg(gemvKernel, 2, sizeof(cl_mem), (void *)&x_in);
    checkErr(status, "Failed to set arg x", "Arg x setted");
    status = clSetKernelArg(gemvKernel, 3, sizeof(double), (void *)&beta);
    checkErr(status, "Failed to set arg beta", "Arg beta setted");
    status = clSetKernelArg(gemvKernel, 4, sizeof(cl_mem), (void *)&y_in);
    checkErr(status, "Failed to set arg y", "Arg y setted");
    status = clSetKernelArg(gemvKernel, 5, sizeof(int), (void *)&num_rows);
    checkErr(status, "Failed to set arg numRows", "Arg numRows setted");
    status = clSetKernelArg(gemvKernel, 6, sizeof(int), (void *)&num_cols);
    checkErr(status, "Failed to set arg numCols", "Arg numCols setted");

    // Launch the kernel - we launch a single work item hence enqueue a task
    //status = clEnqueueTask(queue, kernel, 0, NULL, NULL);
    size_t global_work_size = num_cols;
    status = clEnqueueNDRangeKernel(queue, gemvKernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    checkErr(status, "Failed to launch kernel", "Kernel launched");

    // Wait for command queue to complete pending events
    status = clFinish(queue);
    checkErr(status, "Failed to finish queue", "Queue finished");

    // Copy results from device to host
    status = clEnqueueReadBuffer(queue, y_in, CL_TRUE, 0, sizeof(double) * num_cols, y, 0, NULL, NULL);
    checkErr(status, "Failed to copy data from device", "Data copied from device");

    status = clReleaseMemObject(x_in);
    checkErr(status, "Failed to release x", "x released");
    status = clReleaseMemObject(y_in);
    checkErr(status, "Failed to release y", "y released");
    status = clReleaseMemObject(A_in);
    checkErr(status, "Failed to release A", "A released");
}

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

void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error)
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

    bb = dot(b, b, size);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemv(1.0, A, p, 0.0, Ap, size, size);
        alpha = rr / dot(p, Ap, size);
        axpby(alpha, p, 1.0, x, size);
        axpby(-alpha, Ap, 1.0, r, size);
        rr_new = dot(r, r, size);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / bb) < rel_error) { break; }
        axpby(1.0, r, beta, p, size);
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


void checkErr(cl_int status, std::string error, std::string success = ""){
    if (status != CL_SUCCESS)
        std::cout << "[ERR]: " << error << std::endl;
    else if (DEBUG)
        std::cout << "[INFO]: " << success << std::endl;
}

void cleanup(){
    
}