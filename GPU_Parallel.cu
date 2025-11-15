#include "MatrixGenerator.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <cstdlib>

#define TILE_SIZE 16

void cudaCheckError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void eliminationKernel(Scalar* d_Aug, int N, int M, int pivot_row_idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i < N) {
        if (i != pivot_row_idx) {
            
            Scalar factor = d_Aug[i * M + pivot_row_idx];
            
            for (int j = pivot_row_idx; j < M; ++j) {
                d_Aug[i * M + j] -= factor * d_Aug[pivot_row_idx * M + j];
            }
        }
    }
}

double gaussJordanParallel(int N, vector<Scalar>& Aug) {
    int M = N + 1;
    size_t size = N * M * sizeof(Scalar);
    Scalar *d_Aug;

    cudaCheckError(cudaMalloc((void**)&d_Aug, size)); 
    cudaCheckError(cudaMemcpy(d_Aug, Aug.data(), size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    dim3 threadsPerBlock(TILE_SIZE * TILE_SIZE);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    cudaCheckError(cudaEventRecord(start, 0)); 
    
    for (int k = 0; k < N; ++k) {
        Scalar pivot = Aug[k * M + k];
        if (abs(pivot) < 1e-9) { 
            std::cerr << "Warning: Near-zero pivot detected." << std::endl;
            break;
        }

        for (int j = k; j < M; ++j) {
            Aug[k * M + j] /= pivot;
        }
        
        cudaCheckError(cudaMemcpy(d_Aug + k * M, Aug.data() + k * M, M * sizeof(Scalar), cudaMemcpyHostToDevice));
        
        eliminationKernel<<<numBlocks, threadsPerBlock>>>(d_Aug, N, M, k);
        
        cudaCheckError(cudaDeviceSynchronize()); 
    }
    
    cudaCheckError(cudaEventRecord(stop, 0));
    cudaCheckError(cudaEventSynchronize(stop)); 

    float t_gpu_ms;
    cudaCheckError(cudaEventElapsedTime(&t_gpu_ms, start, stop));

    cudaCheckError(cudaMemcpy(Aug.data(), d_Aug, size, cudaMemcpyDeviceToHost));
    
    cudaCheckError(cudaFree(d_Aug));
    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(stop));

    return (double)t_gpu_ms;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <Matrix Size N>" << endl;
        return 1;
    }
    const int N = atoi(argv[1]);
    const int M = N + 1;
    const int REPEATS = 5;
    
    vector<Scalar> A_original, B_original, X_true;
    generateSystem(N, A_original, B_original, X_true);
    
    double total_time_ms = 0.0;
    Scalar residual = 0.0;

    for (int r = 0; r < REPEATS; ++r) {
        vector<Scalar> Aug(N * M);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                Aug[i * M + j] = A_original[i * N + j];
            }
            Aug[i * M + N] = B_original[i];
        }

        double time_ms = gaussJordanParallel(N, Aug);
        total_time_ms += time_ms;

        vector<Scalar> X_gpu(N);
        for (int i = 0; i < N; ++i) {
            X_gpu[i] = Aug[i * M + N];
        }

        if (r == REPEATS - 1) { 
            residual = calculateResidual(N, A_original, X_gpu, B_original);
        }
    }

    double avg_time_ms = total_time_ms / REPEATS;
    cout << fixed << setprecision(4);
    // Mencetak output dalam format yang mudah diproses: N, Waktu, Residual
    cout << N << "," << avg_time_ms << "," << residual << endl;
    
    return 0;
}