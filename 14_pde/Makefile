10_seq: 10_cavity.cpp
	g++ 10_cavity.cpp
10_openmp: 10_cavity-openmp.cpp
	g++ 10_cavity-openmp.cpp -fopenmp
10_openacc: 10_cavity-openacc.cpp
	nvc++ 10_cavity-openacc.cpp -acc -gpu=managed -Minfo
10_simd: 10_cavity-simd.cpp
	g++ 10_cavity-simd.cpp -march=native -O3
10_cuda: 10_cavity-cuda.cu
	nvcc 10_cavity-cuda.cu
