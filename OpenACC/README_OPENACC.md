# Conjugate Gradient Method with OpenACC on MeluXina

## Overview
This guide provides instructions for running a OpenACC-accelerated program using the algorithm itself on an interactive node.
Additionally, a bash file was provided to further test the implementations on MeluXina.

## Usage Steps for OpenACC

### Running the program in Interactive Mode

#### Step 1: Node Allocation
To access the designated node, use the following command:

```bash
salloc -A p200301 --res gpudev -q dev -N 1 -t 01:00:00
```

#### Step 2: Load Modules
Load the Intel and NVHPC-CUDA module to enable functionality:

```bash
module load intel
module load  NVHPC/23.7-CUDA-12.2.0
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
Compile the OpenACC code using nvcc++:

```bash
nvc++ -fast -Minfo=all -acc src/cg_openacc.cpp -o cg_openacc
```

#### Step 6: Run Code
Run the compiled code with the generated input files:

```bash
./cg_openacc io/matrix.bin io/rhs.bin io/sol.bin
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

#### Step 3: Change into OpenACC directory
Change into the provided OpenACC directory:

```bash
cd OpenACC
```

#### Step 4: Run batch script
Run the provided batch script using the following command:

```bash
sbatch cg_openacc.sh
```

#### Step 5: See output
Once the program is finished you can see the output with the following command:

```bash
cat openacc_<job_id>.out
```