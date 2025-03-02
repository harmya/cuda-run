#include "../cuda_utils.h"


__global__ void simple(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block: %d\nThread:%d\nIndex:%d\n-----------------------------------\n", blockIdx.x , threadIdx.x, i);
}

void initVector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    int n = 8;
    size_t size = n * sizeof(float);

    float* a = (float *) malloc(size);
    initVector(a, n);


    float *d_A;
    checkCudaErrors(cudaMalloc(&d_A, size));

    int threadsPerBlock = 2;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    cudaEvent_t start, stop;
    startTimer(&start, &stop);
    
    simple<<<blocksPerGrid, threadsPerBlock>>>(d_A, n);
    checkCudaErrors(cudaGetLastError());
    
    float milliseconds = stopTimer(start, stop);

    calculatePerformance(n, milliseconds);

    checkCudaErrors(cudaFree(d_A));
    
    free(a);

    return 0;

}