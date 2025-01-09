# CPC 7th China Parallel Application Challenge on Domestic CPU (2023) Preliminary Optimization Report

## Project Overview
This technical report is completed by a team from the University of Chinese Academy of Sciences. For the preliminary round of the 7th China Parallel Application Challenge on Domestic CPU (CPC), we have conducted in-depth optimization on the Preconditioned Conjugate Gradient (PCG) algorithm to enhance its performance in solving sparse linear equation systems.

## Compilation and Execution

### Compilation
To compile the program, please execute the following command in the terminal:
```bash
make
```
This will compile the source code according to the rules defined in the `Makefile`, generating the executable file.

### Execution
After compilation, to run the program, please execute the following command in the terminal:
```bash
./run.sh
```
This will execute the `run.sh` script, launching the program and running the corresponding functionality.

## Result Summary
- **Program Execution Time**: The optimized program ultimately takes 37.7 seconds to run.
- **Speedup Ratio**: The optimized program achieves a speedup ratio of approximately 34 times.
- **Parallel Environment**: Utilizes parallelism within a single core group using coprocessors.

## Optimization Process

### Performance Analysis and Algorithm Understanding
- **Algorithm Background**: The competition problem is based on the Preconditioned Conjugate Gradient algorithm, which solves sparse linear equation systems through iterative steps.
- **Core Operators**: The core operators in the algorithm include sparse matrix-vector multiplication, vector inner product, and element-wise vector operations such as vector scaling and addition. These correspond to steps like residual calculation, direction search, and step length update.
- **Performance Analysis**: More than 80% of the execution time of the baseline program is concentrated on the sparse matrix-vector multiplication operator. We analyzed the non-zero pattern of the sparse matrix to guide subsequent optimization strategies.

### Coprocessor Parallel Optimization
- **Core Operator Porting**: All core operators of the PCG algorithm are ported to run on coprocessors.
- **Sparse Matrix-Vector Multiplication Optimization**: DMA is used to store the matrix and result vector in LDM to accelerate memory access efficiency and avoid direct reads from main memory.
- **Vector Inner Product Optimization**: A two-stage parallel reduction method is employed for acceleration.
- **Format Conversion Optimization**: For the ldu2csr format conversion operation, an array recording the target positions of non-zero elements is generated to enable parallel format conversion.

### Compilation Optimization and Operator Fusion
- **Compilation Optimization Options**: Various coprocessor compilation optimization options were tested. Loop unrolling and fused multiply-add operations were found to reduce the execution time of sparse matrix-vector multiplication by 10-20%; automatic vectorization improved the performance of element-wise operators by nearly 10%.
- **Operator Fusion**: By observing the data flow in the pcg algorithm, operators with data dependencies were fused to avoid interactions with main memory. This increased the data residency rate in high-speed caches, reduced bandwidth pressure, and shortened the overall program execution time. This stage of optimization led to an approximate 2-fold performance improvement.

## Optimization Effect
Through a series of comprehensive optimizations, significant performance improvements were achieved. The program optimization record table displays the performance enhancements after each optimization step, with the final program execution time  
