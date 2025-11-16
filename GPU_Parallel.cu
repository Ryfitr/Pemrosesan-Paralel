#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

typedef double Scalar;

#define BLOCK_SIZE 256

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// Fungsi untuk membaca matriks A|B dari file binary
Scalar* read_matrix(int N, const string& filename) {
    Scalar* Aug = new Scalar[N * (N + 1)];
    ifstream file(filename, ios::binary | ios::in);
    if (!file.is_open()) {
        cerr << "Error: Tidak dapat membaca file " << filename << endl;
        exit(1);
    }
    file.read((char*)Aug, N * (N + 1) * sizeof(Scalar));
    file.close();
    return Aug;
}

// Fungsi untuk membaca vektor X_true dari file binary
Scalar* read_xtrue(int N, const string& filename) {
    Scalar* X_true = new Scalar[N];
    ifstream file(filename, ios::binary | ios::in);
    if (!file.is_open()) {
        cerr << "Error: Tidak dapat membaca file " << filename << endl;
        exit(1);
    }
    file.read((char*)X_true, N * sizeof(Scalar));
    file.close();
    return X_true;
}

// Fungsi untuk menghitung Residual (||X_calc - X_true||)
Scalar calculate_residual(Scalar* final_Aug, Scalar* X_true, int N) {
    Scalar residual_norm = 0.0;
    for (int i = 0; i < N; ++i) {
        Scalar error = final_Aug[i * (N + 1) + N] - X_true[i];
        residual_norm += error * error;
    }
    return std::sqrt(residual_norm);
}

__global__ void elimination_kernel(Scalar* d_Aug, int N, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || row == k) return;
    
    Scalar factor = d_Aug[row * (N + 1) + k];
    
    // ELIMINATE ALL COLUMNS from 0 to N
    for (int col = 0; col < N + 1; ++col) {
        d_Aug[row * (N + 1) + col] -= factor * d_Aug[k * (N + 1) + col];
    }
}

void gauss_jordan_gpu(Scalar* h_Aug, int N) {
    Scalar *d_Aug;
    size_t size = N * (N + 1) * sizeof(Scalar);
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Aug, size));

    int blockSize = BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    for (int k = 0; k < N; ++k) {
        // === SEMUA PIVOTING & NORMALIZATION DI HOST ===
        int pivot_row = k;
        Scalar max_val = std::abs(h_Aug[k * (N + 1) + k]);
        for (int i = k + 1; i < N; ++i) {
            if (std::abs(h_Aug[i * (N + 1) + k]) > max_val) {
                max_val = std::abs(h_Aug[i * (N + 1) + k]);
                pivot_row = i;
            }
        }

        if (pivot_row != k) {
            for (int j = 0; j < N + 1; ++j) {
                Scalar temp = h_Aug[k * (N + 1) + j];
                h_Aug[k * (N + 1) + j] = h_Aug[pivot_row * (N + 1) + j];
                h_Aug[pivot_row * (N + 1) + j] = temp;
            }
        }

        Scalar pivot_val = h_Aug[k * (N + 1) + k];
        if (std::abs(pivot_val) < 1.0e-12) {
            continue;
        }

        // Normalize pivot row
        for (int j = k; j < N + 1; ++j) {
            h_Aug[k * (N + 1) + j] /= pivot_val;
        }

        // === COPY KE DEVICE UNTUK ELIMINATION ===
        CHECK_CUDA_ERROR(cudaMemcpy(d_Aug, h_Aug, size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // === ELIMINATION DI GPU ===
        elimination_kernel<<<numBlocks, blockSize>>>(d_Aug, N, k);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // === COPY BACK KE HOST ===
        CHECK_CUDA_ERROR(cudaMemcpy(h_Aug, d_Aug, size, cudaMemcpyDeviceToHost));
    }
    
    CHECK_CUDA_ERROR(cudaFree(d_Aug));
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <matrix_size_N>" << endl;
        return 1;
    }

    int N = atoi(argv[1]);
    int num_runs = 5;
    double total_time = 0.0;
    Scalar final_residual = 0.0;

    string xtrue_filename = "xtrue_" + to_string(N) + ".bin";
    Scalar* X_true = read_xtrue(N, xtrue_filename);

    for (int run = 0; run < num_runs; ++run) {
        string matrix_filename = "matrix_" + to_string(N) + ".bin";
        Scalar* h_Aug_final = read_matrix(N, matrix_filename);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        gauss_jordan_gpu(h_Aug_final, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;

        final_residual = calculate_residual(h_Aug_final, X_true, N);
        
        if (N == 256 && run == 0) { 
            cout << "\n--- HASIL VEKTOR SOLUSI X (GPU N=256) ---" << endl;
            for (int i = 0; i < N; ++i) {
                if (i < 5 || i >= N - 5) {
                    cout << "X[" << i << "] = " << fixed << setprecision(12) << h_Aug_final[i * (N + 1) + N] << endl;
                } else if (i == 5) {
                    cout << "..." << endl; 
                }
            }
            cout << "Residual Actual: " << fixed << setprecision(12) << final_residual << endl;
            cout << "---------------------------------------" << endl;
        }

        delete[] h_Aug_final;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    delete[] X_true;
    double avg_time = total_time / num_runs;
    cout << fixed << setprecision(4) << N << "," << avg_time << "," << final_residual << endl;

    return 0;
}