#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>


#define checkCudaErrors(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}


inline void initRandom() {
    srand(time(NULL));
}


inline void startTimer(cudaEvent_t* start, cudaEvent_t* stop) {
    checkCudaErrors(cudaEventCreate(start));
    checkCudaErrors(cudaEventCreate(stop));
    checkCudaErrors(cudaEventRecord(*start, NULL));
}


inline float stopTimer(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Execution time: %.4f milliseconds\n", milliseconds);
    
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    
    return milliseconds;
}


inline float calculatePerformance(int operations, float milliseconds) {
    float gigaFlops = (operations / milliseconds) * 1.0e-6;
    printf("Performance: %.4f GFlop/s\n", gigaFlops);
    return gigaFlops;
}

#endif 