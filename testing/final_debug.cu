#include <iostream>
#include <cuda_runtime.h>

using namespace std;

void print_matrix(double* Aug, int N, const char* label) {
    cout << label << ":" << endl;
    for (int i = 0; i < N; i++) {
        cout << "Row " << i << ": ";
        for (int j = 0; j <= N; j++) {
            cout << Aug[i * (N + 1) + j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

__global__ void elimination_kernel_detailed(double* Aug, int N, int k) {
    int row = threadIdx.x;
    if (row >= N || row == k) return;
    
    double factor = Aug[row * (N + 1) + k];
    
    // DEBUG: Print sebelum elimination
    if (row == 1 && k == 0) {
        printf("=== KERNEL DEBUG k=%d, row=%d ===\n", k, row);
        printf("Pivot row %d: ", k);
        for (int c = 0; c <= N; c++) printf("%.2f ", Aug[k * (N + 1) + c]);
        printf("\n");
        printf("Target row %d before: ", row);
        for (int c = 0; c <= N; c++) printf("%.2f ", Aug[row * (N + 1) + c]);
        printf("\n");
        printf("Factor = Aug[%d][%d] = %.2f\n", row, k, factor);
    }
    
    for (int col = 0; col <= N; col++) {
        double pivot_val = Aug[k * (N + 1) + col];
        double old_val = Aug[row * (N + 1) + col];
        Aug[row * (N + 1) + col] = old_val - factor * pivot_val;
        
        if (row == 1 && k == 0 && col == 1) {
            printf("col%d: %.2f = %.2f - (%.2f * %.2f)\n", 
                   col, Aug[row * (N + 1) + col], old_val, factor, pivot_val);
        }
    }
    
    if (row == 1 && k == 0) {
        printf("Target row %d after:  ", row);
        for (int c = 0; c <= N; c++) printf("%.2f ", Aug[row * (N + 1) + c]);
        printf("\n=====================\n");
    }
}

void cpu_reference() {
    cout << "=== CPU REFERENCE IMPLEMENTATION ===" << endl;
    const int N = 3;
    double cpu_Aug[12] = {  // N*(N+1) = 3*4 = 12
        2,  1, -1,  8,
        -3, -1, 2, -11,  
        -2, 1,  2, -3
    };
    
    print_matrix(cpu_Aug, N, "CPU INITIAL");
    
    for (int k = 0; k < N; k++) {
        // Normalize
        double pivot = cpu_Aug[k * (N + 1) + k];
        cout << "Normalize row " << k << ", pivot=" << pivot << endl;
        for (int j = k; j <= N; j++) {
            cpu_Aug[k * (N + 1) + j] /= pivot;
        }
        print_matrix(cpu_Aug, N, "CPU AFTER NORMALIZE");
        
        // Eliminate
        for (int i = 0; i < N; i++) {
            if (i == k) continue;
            double factor = cpu_Aug[i * (N + 1) + k];
            for (int j = 0; j <= N; j++) {
                cpu_Aug[i * (N + 1) + j] -= factor * cpu_Aug[k * (N + 1) + j];
            }
        }
        print_matrix(cpu_Aug, N, "CPU AFTER ELIMINATION");
    }
    
    cout << "CPU FINAL SOLUTION: ";
    for (int i = 0; i < N; i++) {
        cout << "x" << i << "=" << cpu_Aug[i * (N + 1) + N] << " ";
    }
    cout << endl << endl;
}

int main() {
    const int N = 3;
    cpu_reference(); // Run CPU first for reference
    
    double h_Aug[12] = {
        2,  1, -1,  8,
        -3, -1, 2, -11,  
        -2, 1,  2, -3
    };
    
    print_matrix(h_Aug, N, "GPU INITIAL");
    
    double *d_Aug;
    cudaMalloc(&d_Aug, N * (N + 1) * sizeof(double));
    
    for (int k = 0; k < N; k++) {
        cout << "=== GPU ITERATION k=" << k << " ===" << endl;
        
        // Normalize pivot row
        double pivot = h_Aug[k * (N + 1) + k];
        cout << "Normalize row " << k << ", pivot=" << pivot << endl;
        for (int j = k; j <= N; j++) {
            h_Aug[k * (N + 1) + j] /= pivot;
        }
        print_matrix(h_Aug, N, "AFTER NORMALIZE");
        
        // Copy to device
        cudaMemcpy(d_Aug, h_Aug, N * (N + 1) * sizeof(double), cudaMemcpyHostToDevice);
        
        // Elimination on GPU
        cout << "GPU Elimination..." << endl;
        elimination_kernel_detailed<<<1, N>>>(d_Aug, N, k);
        cudaDeviceSynchronize();
        
        // Copy back
        cudaMemcpy(h_Aug, d_Aug, N * (N + 1) * sizeof(double), cudaMemcpyDeviceToHost);
        print_matrix(h_Aug, N, "AFTER GPU ELIMINATION");
    }
    
    cout << "GPU FINAL SOLUTION: ";
    for (int i = 0; i < N; i++) {
        cout << "x" << i << "=" << h_Aug[i * (N + 1) + N] << " ";
    }
    cout << endl;
    
    cudaFree(d_Aug);
    return 0;
}