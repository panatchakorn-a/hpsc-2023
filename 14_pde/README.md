# Final Assignment
### Source code:
- 10_cavity.cpp
- 10_cavity-openmp.cpp
- 10_cavity-openacc.cpp
- 10_cavity-cuda.cpp

### Summary of Results (Runtime, Variables value check)
- results.txt

### Notes on how to compile
Do `module purge` beforehand if needed
- No parallelization (10_cavity.cpp)
    ```
    module load gcc
    make 10_seq
    ./a.out
    ```
- OpenMP
    ```
    module load gcc
    make 10_openmp
    ./a.out
    ```
- OpenACC (cannot do in login node)
    ```
    module load nvhpc/22.2
    make 10_openacc
    ./a.out
    ```
- CUDA (cannot do in login node)
    ```
    module load cuda
    make 10_cuda
    ./a.out
    ```
