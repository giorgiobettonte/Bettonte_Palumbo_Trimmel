# Conjugate Gradient Method with SYCL and Math Kernel Library on MeluXina

## Overview
This guide provides instructions for running a SYCL-accelerated program using the MKL library from oneAPI on an interactive node (CPU).
Additionally, a bash file was provided to further test the implementations on MeluXina.

## Problems
This section describes some of the problem that could not be fixed before the submission deadline.
Most of them deal with either old documentation or incompatible modules which is why these code versions could not be implemented in time.

### SYCL without library on the CPU device
Due to SYCL being a relatively new programming paradigm, some bugs are still present. 
There is currently one concerning the implementation of a simple dot-product, which does not seem to be fixed by now. 
Hence, when we tried to only utilize the written function for the dot-product with SYCL, the program failed. We still provide the code 
in this repository under `SYCL/cd_sycl.cpp`, in case of the bug being corrected.
For more information please refer to
[StackOverflow](https://stackoverflow.com/questions/75621264/sycl-dot-product-code-gives-wrong-results).

### FPGA Implementation
The current [documentation](https://ekieffer.github.io/oneAPI-FPGA/compile/) of running SYCL on a MeluXina FPGA is not up-to-date. 
We tried to run it on the provided emulator and later on the FPGA itself, nevertheless, the compiler and linking step fail due to either the
emulation software itself or problems with the MKL library (does not recognize it).

For completeness purposes you can find the emulation, static report and full compilation in the next sections (in case those issues will be fixed in the future):

#### Emulation

##### Load modules
```bash
module load intel
module load env/release/2021.3
module load intel-compilers
module load ifpga/2021.3.0
module load ifpgasdk/20.4
```

#### Compile for emulation
```bash
icpx -fsycl -fintelfpga -qactypes src/conjugate_gradients_sycl.cpp -o conjugate_gradients_sycl.fpga_emu
```

#### Static report

##### Load modules
```bash
module load intel
module load env/release/2021.3
module load intel-compilers
module load ifpga/2021.3.0
module load ifpgasdk/20.4
```

##### Compile for FPGA early image
```bash
icpx -fsycl -fintelfpga -qactypes -Xshardware -fsycl-link=early -Xstarget=Stratix10 cg_sycl.cpp -o cg_sycl_report.a
```

#### Full compilation

##### Load modules
```bash
module load intel
module load env/release/2021.3
module load intel-compilers
module load 520nmx/20.4
module load ifpgasdk/20.4
```

##### Compile for FPGA early image
```bash
icpx -fsycl -fintelfpga -qactypes -Xshardware -Xstarget=Stratix10 -DFPGA_HARDWARE cg_sycl.cpp -o cg_sycl_report.fpga
```

## Usage Steps for SYCL and the MKL library on the CPU device 

### Running the program in Interactive Mode

#### Step 1: Node Allocation
To access the designated node, use the following command:

```bash
salloc -A p200301 --res cpudev -q dev -N 1 -t 01:00:00
```

#### Step 2: Load Modules
Load the Intel and  module to enable functionality:

```bash
module load intel
module load intel-compilers
module load imkl/2023.1.0
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
Compile the SYCL code using the intel icpx compiler:

```bash
icpx -fsycl -DMKL_ILP64 -I${MKLROOT}/include SYCL/cg_sycl_mkl.cpp -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -ltbb -pthread -ldl -lm -o cg_sycl_mkl
```

#### Step 6: Run Code
Run the compiled code with the generated input files:

```bash
./cg_sycl_mkl io/matrix.bin io/rhs.bin io/sol.bin
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

#### Step 3: Change into SYCL directory
Change into the provided OpenACC directory:

```bash
cd SYCL
```

#### Step 4: Run batch script
Run the provided batch script using the following command:

```bash
sbatch cg_sycl_mkl.sh
```

#### Step 5: See output
Once the program is finished you can see the output with the following command:

```bash
cat sycl_mkl_<job_id>.out
```