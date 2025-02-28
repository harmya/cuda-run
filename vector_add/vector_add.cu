#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define checkCudaErrors(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void initVector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

void verifyResult(const float *A, const float *B, const float *C, int n) {
    for (int i = 0; i < n; i++) {
        float expected = A[i] + B[i];
        if (fabs(C[i] - expected) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            fprintf(stderr, "Expected: %f, Got: %f\n", expected, C[i]);
            exit(EXIT_FAILURE);
        }
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    srand(time(NULL));

    float *h_A = (float*) malloc(size);
    float *h_B = (float*) malloc(size);
    float *h_C = (float*) malloc(size);

    initVector(h_A, n);
    initVector(h_B, n);

    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(&d_A, size));
    checkCudaErrors(cudaMalloc(&d_B, size));
    checkCudaErrors(cudaMalloc(&d_C, size));

    checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, NULL));
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Execution time: %.4f milliseconds\n", milliseconds);

    checkCudaErrors(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    verifyResult(h_A, h_B, h_C, n);
    
    float gigaFlops = (n / milliseconds) * 1.0e-6; 
    printf("Performance: %.4f GFlop/s\n", gigaFlops);

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    
    free(h_A);
    free(h_B);
    free(h_C);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return 0;

}