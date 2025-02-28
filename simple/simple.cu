#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void accessThread() {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread: [blockIdx.x=%d, threadIdx.x=%d] - Global ID: %d\n", blockIdx.x, threadIdx.x, threadId);
}

int main() {
    int threads_per_block = 4;
    int num_blocks_per_grid = 4;

    printf("CUDA Kernel Launch with %d blocks of %d threads\n", num_blocks_per_grid, threads_per_block);

    accessThread<<<num_blocks_per_grid, threads_per_block>>>();

    cudaError_t err = cudaDeviceSynchronize();
    checkCudaError(err, "Kernel Launch Failed");

    err = cudaGetLastError();
    checkCudaError(err, "Failed to execute kernel");

    printf("Done!\n");
    return 0;
}