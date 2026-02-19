#include <cuda_runtime_api.h>
#include <stdio.h>
int main() { 
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if(err != cudaSuccess) printf("Error: %d\n", err);
    else printf("CUDA Devices: %d\n", count);
    return 0;
}
