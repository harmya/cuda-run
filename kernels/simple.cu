#include "../cuda_utils.h"


__global__ void simple(int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        printf("Block: %d\nThread:%d\nIndex:%d\nValue:%d\n-----------------------------------\n", blockIdx.x , threadIdx.x, i, a[i]);
    }
}

void initVector(int *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (int)(((float)rand() / RAND_MAX) * 10);
    }
}

int main() {
    int n = 8;
    size_t size = n * sizeof(int);

    int* a = (int *) malloc(size);
    initVector(a, n);

    for (int i = 0; i < n; i++) {
        printf("a[%d] = %d\n", i, a[i]);
    }

    int *d_A;
    checkCudaErrors(cudaMalloc(&d_A, size));

    checkCudaErrors(cudaMemcpy(d_A, a, size, cudaMemcpyHostToDevice));

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