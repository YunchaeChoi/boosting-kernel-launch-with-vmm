#include <stdio.h>
#include <stdlib.h>
#include <low.h>

int main() {
    cudaFree(0);

    size_t size = 16384 * 16384 * 4; // 1 GiB
    float* d_A;
    cudaMalloc(&d_A, size);
    cudaError_t err = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(err));
    vmm_struct d_B =  low_level_allocation(size);

    cudaEvent_t start;
    cudaEvent_t end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    low_level_free(d_B);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, end);

    printf("VMM: %fms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_A);
    return 0;
}

