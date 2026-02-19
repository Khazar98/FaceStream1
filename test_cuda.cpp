#include <cuda_runtime.h>
#include <iostream>

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << " (" << (int)err << ")" << std::endl;
        return 1;
    }
    std::cout << "CUDA Device Count: " << count << std::endl;
    return 0;
}
