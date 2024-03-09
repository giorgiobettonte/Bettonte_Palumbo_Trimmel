// OpenCL implementation of basic Linear Algebra operations
// used to acelerate Conjugate Gradient on a FPGA

// Dot product
__kernel void dot(__global double * restrict x, 
                  __global double * restrict y,
                  __global double * restrict z,
                  int dim,
                  int numElem) 
{
    int id = get_global_id(0);

    // Check boundary
    if (id >= dim)
        return;
    
    // Compute a[dot]b
    int offset = id * numElem;
    
    #pragma unroll
    for(int i = offset; i < offset + numElem; i++)
        z[id] += x[i] * y[i];
}

// axpby
__kernel void axpby(double alpha,
                    __global double * restrict x,
                    double beta,
                    __global double * restrict y,
                    int dim)
{
    int id = get_global_id(0);

    // Check boundary
    if (id >= dim)
        return;

    y[id] = alpha * x[id] + beta * y[id];   
}

// gemv
__kernel void gemv(double alpha,
                   __global double * restrict A,
                   __global double * restrict x,
                   double beta,
                   __global double * restrict y,
                   int num_rows,
                   int num_cols)
{
    int id = get_global_id(0);
    double y_val = 0.0;

    #pragma unroll
    for(int i = 0; i < num_cols; i++){
        y_val += alpha * A[id * num_cols + i] * x[i];
    }

    y[id] = beta * y[id] + y_val;
}