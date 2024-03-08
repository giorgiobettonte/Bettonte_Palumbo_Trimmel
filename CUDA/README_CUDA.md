# Conjugate Gradient Method with CUDA and cuBLAS on MeluXina

## Overview
This guide provides instructions for running a CUDA-accelerated program using either the algorithm itself or the cuBLAS library on an interactive node. 
Additionally, two bash file were provided to further test the implementations on MeluXina.

## Usage Steps for CUDA

### Running the program in Interactive Mode

#### Step 1: Node Allocation
To access the designated node, use the following command:

```bash
salloc -A p200301 --res gpudev -q dev -N 1 -t 01:00:00
```

#### Step 2: Load Modules
Load the Intel and CUDA module to enable functionality:

```bash
module load intel
module load CUDA/12.2.0
```

#### Step 3: Prepare Input/Output Directory
Create a directory to store input and output files:

```bash
mkdir io
```

#### Step 4: Generate Matrix System
Generate the matrix system to solve by executing. 
Choose size according to your preferences:

```bash
./random_spd_system.sh <matrix_size> io/matrix.bin io/rhs.bin
```

#### Step 5: Compile Code
Compile the CUDA code using nvcc:

```bash
nvcc -O2 CUDA/cg_cuda.cu -o cg_cuda
```

#### Step 6: Run Code
Run the compiled code with the generated input files:

```bash
./cg_cuda io/matrix.bin io/rhs.bin io/sol.bin
```

### Running the program in Batch Mode

#### Step 1: Prepare Input/Output Directory
Create a directory to store input and output files:

```bash
mkdir io
```

#### Step 2: Generate Matrix System
Generate the matrix system to solve by executing.
Choose size according to your preferences:

```bash
./random_spd_system.sh <matrix_size> io/matrix.bin io/rhs.bin
```

#### Step 3: Change into CUDA directory
Change into the provided CUDA directory:

```bash
cd CUDA
```

#### Step 4: Run batch script
Run the provided batch script using the following command:

```bash
sbatch cg_cuda.sh
```

#### Step 5: See output
Once the program is finished you can see the output with the following command:

```bash
cat cuda_<job_id>.out
```

## Usage Steps for CUDA with cuBLAS

### Running the program in Interactive Mode

#### Step 1: Node Allocation
To access the designated node, use the following command:

```bash
salloc -A p200301 --res gpudev -q dev -N 1 -t 01:00:00
```

#### Step 2: Load Modules
Load the Intel and CUDA module to enable functionality:

```bash
module load intel
module load CUDA/12.2.0
```

#### Step 3: Prepare Input/Output Directory
Create a directory to store input and output files:

```bash
mkdir io
```

#### Step 4: Generate Matrix System
Generate the matrix system to solve by executing.
Choose size according to your preferences:

```bash
./random_spd_system.sh <matrix_size> io/matrix.bin io/rhs.bin
```

#### Step 5: Compile Code
Compile the CUDA code using nvcc:

```bash
nvcc -O2 CUDA/cg_cuda_cublas.cu -o cg_cuda_cublas -lcublas
```

#### Step 6: Run Code
Run the compiled code with the generated input files:

```bash
./cg_cuda_cublas io/matrix.bin io/rhs.bin io/sol.bin
```

### Running the program in Batch Mode

#### Step 1: Prepare Input/Output Directory
Create a directory to store input and output files:

```bash
mkdir io
```

#### Step 2: Generate Matrix System
Generate the matrix system to solve by executing.
Choose size according to your preferences:

```bash
./random_spd_system.sh <matrix_size> io/matrix.bin io/rhs.bin
```

#### Step 3: Change into CUDA directory
Change into the provided CUDA directory:

```bash
cd CUDA
```

#### Step 4: Run batch script
Run the provided batch script using the following command:

```bash
sbatch cg_cuda_cublas.sh
```

#### Step 5: See output
Once the program is finished you can see the output with the following command:

```bash
cat cuda_cublas_<job_id>.out
```

