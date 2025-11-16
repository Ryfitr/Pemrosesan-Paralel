#include <iostream>
#include <cuda_runtime.h>

using namespace std;

void print_matrix(double* Aug, int N, const char* label) {
    cout << "\n" << label << ":" << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= N; j++) {
            cout << Aug[i * (N + 1) + j] << "\t";
        }
        cout << endl;
    }
}

__global__ void elimination_kernel_debug(double* Aug, int N, int k) {
    int row = threadIdx.x;
    if (row >= N || row == k) return;
    
    double factor = Aug[row * (N + 1) + k];
    printf("GPU: row=%d, factor=%f\n", row, factor);
    
    // ✅ PASTIKAN line ini: col = 0, bukan col = k
    for (int col = 0; col <= N; col++) {
        Aug[row * (N + 1) + col] -= factor * Aug[k * (N + 1) + col];
    }
}

int main() {
    const int N = 3;
    
    double h_Aug[N * (N + 1)] = {
        2,  1, -1,  8,
        -3, -1, 2, -11,  
        -2, 1,  2, -3
    };
    
    print_matrix(h_Aug, N, "INITIAL MATRIX");
    
    double *d_Aug;
    cudaMalloc(&d_Aug, N * (N + 1) * sizeof(double));
    
    cout << "\n=== ITERATION k=0 ===" << endl;
    double pivot = h_Aug[0 * (N + 1) + 0];
    cout << "Normalizing row 0, pivot=" << pivot << endl;
    for (int j = 0; j <= N; j++) {
        h_Aug[0 * (N + 1) + j] /= pivot;
    }
    print_matrix(h_Aug, N, "AFTER NORMALIZE ROW 0");
    
    cudaMemcpy(d_Aug, h_Aug, N * (N + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cout << "Running GPU elimination for k=0..." << endl;
    elimination_kernel_debug<<<1, N>>>(d_Aug, N, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Aug, d_Aug, N * (N + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    print_matrix(h_Aug, N, "AFTER ELIMINATION k=0");
    
    cout << "\n=== ITERATION k=1 ===" << endl;
    pivot = h_Aug[1 * (N + 1) + 1];
    cout << "Normalizing row 1, pivot=" << pivot << endl;
    for (int j = 1; j <= N; j++) {
        h_Aug[1 * (N + 1) + j] /= pivot;
    }
    print_matrix(h_Aug, N, "AFTER NORMALIZE ROW 1");
    
    cudaMemcpy(d_Aug, h_Aug, N * (N + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cout << "Running GPU elimination for k=1..." << endl;
    elimination_kernel_debug<<<1, N>>>(d_Aug, N, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Aug, d_Aug, N * (N + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    print_matrix(h_Aug, N, "AFTER ELIMINATION k=1");
    
    cout << "\n=== ITERATION k=2 ===" << endl;
    pivot = h_Aug[2 * (N + 1) + 2];
    cout << "Normalizing row 2, pivot=" << pivot << endl;
    for (int j = 2; j <= N; j++) {
        h_Aug[2 * (N + 1) + j] /= pivot;
    }
    print_matrix(h_Aug, N, "AFTER NORMALIZE ROW 2");
    
    cout << "\n=== FINAL SOLUTION ===" << endl;
    for (int i = 0; i < N; i++) {
        cout << "x" << i << " = " << h_Aug[i * (N + 1) + N] << endl;
    }
    cout << "Expected: x0=2, x1=3, x2=-1" << endl;
    
    cudaFree(d_Aug);
    return 0;
}