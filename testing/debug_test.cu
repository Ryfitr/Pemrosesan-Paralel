#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

__global__ void simple_kernel(double* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0;  // Simple operation
    }
}

int main() {
    const int N = 10;
    double h_data[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double *d_data;
    
    std::cout << "Original: ";
    for(int i = 0; i < N; i++) std::cout << h_data[i] << " ";
    std::cout << std::endl;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, N * sizeof(double), cudaMemcpyHostToDevice));
    
    simple_kernel<<<1, N>>>(d_data, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_data, d_data, N * sizeof(double), cudaMemcpyDeviceToHost));
    
    std::cout << "After GPU: ";
    for(int i = 0; i < N; i++) std::cout << h_data[i] << " ";
    std::cout << std::endl;
    
    cudaFree(d_data);
    return 0;
}