#include <iostream>
#include <fstream>
#include <cmath>
#include <CL/sycl.hpp>
#include <thread>

//#include <sycl/ext/intel/fpga_extensions.hpp> // Header for FPGA use

using namespace cl::sycl;


bool read_matrix_from_file(const char *filename, double **matrix_out, size_t &num_rows_out, size_t &num_cols_out, queue &q) {
    double *matrix;
    size_t num_rows;
    size_t num_cols;

    FILE *file = fopen(filename, "rb");
    if (file == nullptr) {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);

    // Allocate memory for matrix on host and device
    matrix = malloc_shared<double>(num_rows * num_cols, q);

    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    num_rows_out = num_rows;
    num_cols_out = num_cols;

    fclose(file);

    return true;
}

bool write_matrix_to_file(const char *filename, const double *matrix, size_t num_rows, size_t num_cols) {
    FILE *file = fopen(filename, "wb");
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

void print_matrix(const double *matrix, size_t num_rows, size_t num_cols, FILE *file = stdout) {
    fprintf(file, "%zu %zu\n", num_rows, num_cols);
    for (size_t r = 0; r < num_rows; r++) {
        for (size_t c = 0; c < num_cols; c++) {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}

// SYCL kernel for dot product
void dot(const double *x, const double *y, double *result, size_t size, queue &q) {
    // Create buffer
    buffer<double, 1> buf_x(x, range<1>(size));
    buffer<double, 1> buf_y(y, range<1>(size));
    buffer<double, 1> buf_result(result, range<1>(1));

    // Submit dot product in queue
    q.submit([&](handler &h) {
        // Create accessors
        auto accessor_x = buf_x.get_access<access::mode::read>(h);
        auto accessor_y = buf_y.get_access<access::mode::read>(h);
        auto accessor_result = buf_result.get_access<access::mode::discard_write>(h); // Changed access mode

        // Perform dot product in parallel
        h.parallel_for(range<1>(size), [=](id<1> idx) {
            accessor_result[0] += accessor_x[idx] * accessor_y[idx];
        });
    }).wait();
}

// SYCL kernel for gemv operation
void gemv(double alpha, const double *A, const double *x, double beta, double *y, size_t num_rows, size_t num_cols, queue &q) {
    // Create buffer
    buffer<double, 1> buf_A(A, range<1>(num_rows * num_cols));
    buffer<double, 1> buf_x(x, range<1>(num_cols));
    buffer<double, 1> buf_y(y, range<1>(num_rows));

    // Submit gemv in queue
    q.submit([&](handler &h) {
        // Create accessors
        auto accessor_A = buf_A.get_access<access::mode::read>(h);
        auto accessor_x = buf_x.get_access<access::mode::read>(h);
        auto accessor_y = buf_y.get_access<access::mode::read_write>(h);

        // Perform gemv in parallel
        h.parallel_for(range<1>(num_rows), [=](id<1> idx) {
            double y_val = 0.0;

            for (size_t c = 0; c < num_cols; c++) {
                y_val += alpha * accessor_A[idx * num_cols + c] * accessor_x[c];
            }

            accessor_y[idx] = beta * accessor_y[idx] + y_val;
        });
    }).wait();
}

// SYCL kernel for axpby operation
void axpby(double alpha, const double *x, double beta, double *y, size_t size, queue &q) {
    // Create buffer
    buffer<double, 1> buf_x(x, range<1>(size));
    buffer<double, 1> buf_y(y, range<1>(size));

    // Submit axpby in queue
    q.submit([&](handler &h) {
        auto accessor_x = buf_x.get_access<access::mode::read>(h);
        auto accessor_y = buf_y.get_access<access::mode::read_write>(h);

        // Perform axpby in parallel
        h.parallel_for(range<1>(size), [=](id<1> idx) {
            accessor_y[idx] = alpha * accessor_x[idx] + beta * accessor_y[idx];
        });
    }).wait();
}

// SYCL kernel for conjugate gradients
void conjugate_gradients(const double *A, const double *b, double *x, size_t size, int max_iters, double rel_error, queue &q) {
    // Define variables
    int num_iters;
    double alpha, beta, pAp, bb, rr, rr_new;

    // Allocate memory for vectors r, p, and Ap on host and device
    double *r =  malloc_shared<double>(size, q);
    double *p = malloc_shared<double>(size, q);
    double *Ap = malloc_shared<double>(size, q);

    // Fill arrays with initial values
    q.fill(x, 0.0, size).wait();;
    q.memcpy(r, b, sizeof(double) * size).wait();;
    q.memcpy(p, b, sizeof(double) * size).wait();

    // Compute dot product
    dot(b, b, &bb, size, q);
    q.wait();

    rr = bb;

    for (num_iters = 1; num_iters <= max_iters; num_iters++) {
        // Compute gemv
        gemv(1.0, A, p, 0.0, Ap, size, size, q);
        q.wait();

        // Compute dot product
        dot(p, Ap, &pAp, size, q);
        q.wait();

        // Calculate new alpha
        alpha = rr / pAp;

        // Compute axpby
        axpby(alpha, p, 1.0, x, size, q);
        axpby(-alpha, Ap, 1.0, r, size, q);
        q.wait();

        // Compute dot product
        dot(r, r, &rr_new, size, q);
        q.wait();

        // Calculate new beta
        beta = rr_new / rr;
        rr = rr_new;

        // Check for convergence
        if (std::sqrt(rr / bb) < rel_error) break;

        // Compute axpby
        axpby(1.0, r, beta, p, size, q);
        q.wait();
    }

    if(num_iters <= max_iters)
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    else
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));

    free(r, q);
    free(p, q);
    free(Ap, q);
}

int main(int argc, char **argv) {
    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");

    const char *input_file_matrix = "io/matrix.bin";
    const char *input_file_rhs = "io/rhs.bin";
    const char *output_file_sol = "io/sol.bin";
    int max_iters = 1000;
    double rel_error = 1e-9;

    if (argc > 1)
        input_file_matrix = argv[1];
    if (argc > 2)
        input_file_rhs = argv[2];
    if (argc > 3)
        output_file_sol = argv[3];
    if (argc > 4)
        max_iters = atoi(argv[4]);
    if (argc > 5)
        rel_error = atof(argv[5]);

    printf("Command line arguments:\n");
    printf("  input_file_matrix: %s\n", input_file_matrix);
    printf("  input_file_rhs:    %s\n", input_file_rhs);
    printf("  output_file_sol:   %s\n", output_file_sol);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("\n");

/* SYCL using FPGA - currently not compatible with MeluXina FPGA
 *
    #if FPGA_SIMULATOR              // simulator device
        auto selector = sycl::ext::intel::fpga_simulator_selector_v;
    #elif FPGA_HARDWARE             // FPGA device (a real FPGA)
        auto selector = sycl::ext::intel::fpga_selector_v;
    #else  // #if FPGA_EMULATOR     // FPGA emulator device (CPU emulation of the FPGA)
        auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    #endif

        // Create the device queue
        sycl::queue q(selector);
*/

    // Print each platform and device
    printf("Possible devices:\n");
    for (const auto &platform : platform::get_platforms()) {
        printf("Platform: %s\n", platform.get_info<info::platform::name>().c_str());

        // Get all devices associated with the platform
        auto devices = platform.get_devices();

        // Iterate over each device
        for (const auto &device : devices) {
            printf("Device: %s (vendor: %s)\n", device.get_info<info::device::name>().c_str(), device.get_info<info::device::vendor>().c_str());
        }
    }

    // Define queue
    queue q;

    // Get device
    auto device = q.get_device();
    printf("Running on device: %s\n", device.get_info<cl::sycl::info::device::name>().c_str());

    // Define variables
    double *matrix;
    double *rhs;
    size_t size;

    {
        printf("Reading matrix from file ...\n");
        size_t matrix_rows;
        size_t matrix_cols;
        bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, matrix_rows, matrix_cols, q);
        if (!success_read_matrix) {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }
        printf("Done\n");
        printf("\n");

        printf("Reading right hand side from file ...\n");
        size_t rhs_rows;
        size_t rhs_cols;
        bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, rhs_rows, rhs_cols, q);
        if (!success_read_rhs) {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }
        printf("Done\n");
        printf("\n");

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

    printf("Solving the system ...\n");

    // Allocate memory for sol on host and device
    double *sol = malloc_shared<double>(size, q);

    conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error, q);

    printf("Done\n");
    printf("\n");

    printf("Writing solution to file ...\n");
    bool success_write_sol = write_matrix_to_file(output_file_sol, sol, size, 1);
    if (!success_write_sol) {
        fprintf(stderr, "Failed to save solution\n");
        return 6;
    }
    printf("Done\n");
    printf("\n");

    free(matrix, q);
    free(rhs, q);
    free(sol, q);

    printf("Finished successfully\n");

    return 0;
}
