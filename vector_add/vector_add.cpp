#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

void initVector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

void vectorAddCPU(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
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

double getCurrentTimeMs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

int main(int argc, char **argv) {
    int n = 1000000;
    size_t size = n * sizeof(float);
    
    srand(time(NULL));
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }
    initVector(h_A, n);
    initVector(h_B, n);
    
    double start_time = getCurrentTimeMs();
    
    vectorAddCPU(h_A, h_B, h_C, n);
    
    double end_time = getCurrentTimeMs();
    double elapsed_time = end_time - start_time;
    
    printf("Execution time: %.4f milliseconds\n", elapsed_time);
    
    verifyResult(h_A, h_B, h_C, n);
    float gigaFlops = (n / elapsed_time) * 1.0e-6;
    printf("Performance: %.4f GFlop/s\n", gigaFlops);

    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}