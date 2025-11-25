#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>
#include <cstdio>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

typedef float Scalar;

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t _e = (err); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif

#define THREADS_PIVOT 1024
#define THREADS_COL   512

static inline Scalar* read_matrix(int N, const string& filename) {
    Scalar* Aug = new Scalar[(size_t)N * (N + 1)];
    ifstream f(filename, ios::binary);
    if (!f.is_open()) {
        cerr << "Error: gagal baca " << filename << endl;
        exit(1);
    }
    f.read(reinterpret_cast<char*>(Aug), (streamsize)((size_t)N * (N + 1) * sizeof(Scalar)));
    f.close();
    return Aug;
}

static inline Scalar* read_xtrue(int N, const string& filename) {
    Scalar* X = new Scalar[N];
    ifstream f(filename, ios::binary);
    if (!f.is_open()) {
        cerr << "Error: gagal baca " << filename << endl;
        exit(1);
    }
    f.read(reinterpret_cast<char*>(X), (streamsize)((size_t)N * sizeof(Scalar)));
    f.close();
    return X;
}

// Menghitung norm error solusi: ||x_computed - Xtrue||_2
static inline Scalar calc_residual(const Scalar* Aug, const Scalar* Xtrue, int N) {
    const long long stride = (long long)(N + 1);
    long double acc = 0.0L;
    for (int i = 0; i < N; ++i) {
        long double e = (long double)Aug[(long long)i * stride + N] - (long double)Xtrue[i];
        acc += e * e;
    }
    return (Scalar)std::sqrt((double)acc);
}

__global__ void find_pivot_kernel(const Scalar* __restrict__ d_Aug,
                                  int N, int k,
                                  int* __restrict__ d_pivot_idx,
                                  Scalar eps) {
    extern __shared__ unsigned char smem[];
    Scalar* s_val = (Scalar*)smem;
    int*    s_idx = (int*)&s_val[blockDim.x];

    const long long pitch = (long long)(N + 1);
    Scalar best = 0.0f;
    int    best_i = -1;

    // Tiap thread cek beberapa baris
    for (int i = k + threadIdx.x; i < N; i += blockDim.x) {
        Scalar v = fabsf(d_Aug[(long long)i * pitch + k]);
        if (v > best) {
            best = v;
            best_i = i;
        }
    }

    s_val[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();

    // Reduction maksimum di shared memory
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_val[threadIdx.x + stride] > s_val[threadIdx.x]) {
                s_val[threadIdx.x] = s_val[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *d_pivot_idx = (s_val[0] < eps) ? -1 : s_idx[0];
    }
}

__global__ void swap_rows_kernel(Scalar* __restrict__ d_Aug, int N, int r1, int r2) {
    if (r1 == r2) return;
    const long long pitch = (long long)(N + 1);
    const long long b1 = (long long)r1 * pitch;
    const long long b2 = (long long)r2 * pitch;

    for (int j = threadIdx.x; j <= N; j += blockDim.x) {
        Scalar tmp      = d_Aug[b1 + j];
        d_Aug[b1 + j]   = d_Aug[b2 + j];
        d_Aug[b2 + j]   = tmp;
    }
}

__global__ void normalize_pivot_row_kernel(Scalar* __restrict__ d_Aug, int N, int k, Scalar pivot_min) {
    const long long pitch = (long long)(N + 1);
    const long long base  = (long long)k * pitch;
    Scalar pivot = d_Aug[base + k];
    if (fabsf(pivot) < pivot_min) return;

    for (int j = k + threadIdx.x; j <= N; j += blockDim.x) {
        d_Aug[base + j] /= pivot;
    }
}

__global__ void eliminate_rows_kernel(Scalar* __restrict__ d_Aug, int N, int k) {
    int row = blockIdx.x;
    if (row == k || row >= N) return;

    const long long pitch   = (long long)(N + 1);
    const long long baseRow = (long long)row * pitch;
    const long long baseK   = (long long)k   * pitch;

    __shared__ Scalar factor;
    if (threadIdx.x == 0) {
        factor = d_Aug[baseRow + k];
    }
    __syncthreads();

    for (int j = k + threadIdx.x; j <= N; j += blockDim.x) {
        d_Aug[baseRow + j] -= factor * d_Aug[baseK + j];
    }
}

static inline bool gauss_jordan_gpu(Scalar* d_Aug, int N, cudaStream_t stream = 0) {
    int* d_pivot = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pivot, sizeof(int)));

    dim3 gridRow(N), blockCol(THREADS_COL);

    for (int k = 0; k < N; ++k) {
        size_t smem = (size_t)THREADS_PIVOT * (sizeof(Scalar) + sizeof(int));

        // Cari pivot di GPU
        find_pivot_kernel<<<1, THREADS_PIVOT, smem, stream>>>(d_Aug, N, k, d_pivot, (Scalar)1.0e-6f);
        CHECK_CUDA_ERROR(cudaGetLastError());

        int h_pivot = -1;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(&h_pivot, d_pivot, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        if (h_pivot < 0) {
            cudaFree(d_pivot);
            return false;
        }

        // Tukar baris k dengan baris pivot
        swap_rows_kernel<<<1, THREADS_COL, 0, stream>>>(d_Aug, N, k, h_pivot);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Normalisasi pivot row
        normalize_pivot_row_kernel<<<1, THREADS_COL, 0, stream>>>(d_Aug, N, k, (Scalar)1.0e-20f);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Eliminasi baris lain
        eliminate_rows_kernel<<<gridRow, blockCol, 0, stream>>>(d_Aug, N, k);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    CHECK_CUDA_ERROR(cudaFree(d_pivot));
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <N> <ID>\n";
        return 1;
    }

    int N  = atoi(argv[1]);
    int ID = atoi(argv[2]);

    string suffix = to_string(N) + "_" + to_string(ID);
    const string mx = "matrix_" + suffix + ".bin";
    const string xt = "xtrue_"  + suffix + ".bin";

    Scalar* Xtrue = read_xtrue(N, xt);

    Scalar* d_Aug = nullptr;
    size_t bytesAug = (size_t)N * (N + 1) * sizeof(Scalar);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Aug, bytesAug));

    // Satu run saja per matriks
    Scalar* h_Aug = read_matrix(N, mx);
    CHECK_CUDA_ERROR(cudaMemcpy(d_Aug, h_Aug, bytesAug, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    bool ok = gauss_jordan_gpu(d_Aug, N);
    if (!ok) {
        cerr << "Peringatan: singular/ill-conditioned terdeteksi.\n";
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA_ERROR(cudaMemcpy(h_Aug, d_Aug, bytesAug, cudaMemcpyDeviceToHost));
    Scalar last_res = calc_residual(h_Aug, Xtrue, N);

    // Output detail solusi ke terminal (hanya N=256, ID=1)
    if (N == 256 && ID == 1) {
        cout << "\n--- HASIL VEKTOR SOLUSI X (GPU N=" << N << ", ID=" << ID << ") ---\n";
        cout.setf(ios::fixed);
        cout << setprecision(12);
        for (int i = 0; i < N; ++i) {
            if (i < 5 || i >= N - 5) {
                cout << "X[" << i << "] = " << h_Aug[(size_t)i * (N + 1) + N] << "\n";
            } else if (i == 5) {
                cout << "...\n";
            }
        }
        cout << "Residual Actual: " << last_res << "\n";
        cout << "---------------------------------------\n";
    }

    delete[] h_Aug;
    delete[] Xtrue;
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_Aug));

    // Cetak baris data ke terminal
    cout.setf(ios::fixed);
    cout << setprecision(4);
    cout << N << "," << ID << "," << ms << "," << last_res << "\n";

    // Tulis juga ke file CSV
    string csv_name = "gpu_results.csv";
    ios_base::openmode mode;

    // Untuk N=256 & ID=1, overwrite dan tulis header
    if (N == 256 && ID == 1) {
        mode = ios::out;
    } else {
        mode = ios::out | ios::app;
    }

    ofstream csv(csv_name, mode);
    if (csv.is_open()) {
        if (N == 256 && ID == 1) {
            csv << "Ukuran,ID,Time,Residual\n";
        }
        csv.setf(ios::fixed);
        csv << setprecision(4);
        csv << N << "," << ID << "," << ms << "," << last_res << "\n";
        csv.close();
    } else {
        cerr << "Peringatan: tidak bisa membuka " << csv_name << " untuk menulis.\n";
    }

    return 0;
}
