#include "MatrixGenerator.h"
#include <cmath>
#include <iomanip>
#include <cstdlib>

void gaussJordanSequential(int N, vector<Scalar>& Aug) {
    int M = N + 1;

    for (int k = 0; k < N; ++k) {
        
        Scalar pivot = Aug[k * M + k];
        if (abs(pivot) < 1e-9) {
            // Error ini menunjukkan kegagalan numerik tanpa pivoting.
            return;
        }

        for (int j = k; j < M; ++j) {
            Aug[k * M + j] /= pivot;
        }

        for (int i = 0; i < N; ++i) {
            if (i != k) {
                Scalar factor = Aug[i * M + k];
                for (int j = k; j < M; ++j) {
                    Aug[i * M + j] -= factor * Aug[k * M + j];
                }
            }
        }
    }
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

        auto start = chrono::high_resolution_clock::now();
        
        gaussJordanSequential(N, Aug);
        
        auto end = chrono::high_resolution_clock::now();
        double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
        total_time_ms += time_ms;

        vector<Scalar> X_cpu(N);
        for (int i = 0; i < N; ++i) {
            X_cpu[i] = Aug[i * M + N];
        }

        if (r == REPEATS - 1) {
            residual = calculateResidual(N, A_original, X_cpu, B_original);
        }
    }

    double avg_time_ms = total_time_ms / REPEATS;
    cout << fixed << setprecision(4);
    // Mencetak output dalam format yang mudah diproses: N, Waktu, Residual
    cout << N << "," << avg_time_ms << "," << residual << endl;    
    return 0;
}