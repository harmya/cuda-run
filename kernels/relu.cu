#include "../cuda_utils.h"

__global__ void matrixReLU(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        int bits = __float_as_int(x);
        int mask = ~(bits >> 31);  
        bits &= mask;  
        output[idx] = __int_as_float(bits);
    }
}

void initMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    }
}

void verifyResult(const float *input, const float *output, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        float expected = input[i] > 0 ? input[i] : 0;
        if (fabs(output[i] - expected) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            fprintf(stderr, "Expected: %f, Got: %f\n", expected, output[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("Result verification successful!\n");
}

int main() {
    int rows = 1000;
    int cols = 1000;
    int matrixSize = rows * cols;
    size_t bytes = matrixSize * sizeof(float);

    initRandom();

    float *h_input = (float*) malloc(bytes);
    float *h_output = (float*) malloc(bytes);

    initMatrix(h_input, rows, cols);

    float *d_input, *d_output;
    checkCudaErrors(cudaMalloc(&d_input, bytes));
    checkCudaErrors(cudaMalloc(&d_output, bytes));

    checkCudaErrors(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 1024;
    int blocksPerGrid = (problemSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    startTimer(&start, &stop);
    
    matrixReLU<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, matrixSize);
    checkCudaErrors(cudaGetLastError());
    
    float milliseconds = stopTimer(start, stop);

    checkCudaErrors(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    verifyResult(h_input, h_output, rows, cols);
    calculatePerformance(matrixSize, milliseconds);

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
    
    free(h_input);
    free(h_output);

    return 0;
}