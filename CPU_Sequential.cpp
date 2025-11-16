#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <string>

using namespace std;

typedef float Scalar;

static inline Scalar* read_matrix(int N, const string& filename) {
    Scalar* Aug = new Scalar[N * (N + 1)];
    ifstream file(filename, ios::binary | ios::in);
    if (!file.is_open()) { cerr << "Error: Tidak dapat membaca " << filename << endl; exit(1); }
    file.read((char*)Aug, (streamsize)(N * (N + 1) * sizeof(Scalar)));
    file.close();
    return Aug;
}

static inline Scalar* read_xtrue(int N, const string& filename) {
    Scalar* X_true = new Scalar[N];
    ifstream file(filename, ios::binary | ios::in);
    if (!file.is_open()) { cerr << "Error: Tidak dapat membaca " << filename << endl; exit(1); }
    file.read((char*)X_true, (streamsize)(N * sizeof(Scalar)));
    file.close();
    return X_true;
}

static inline Scalar calculate_residual(Scalar* final_Aug, Scalar* X_true, int N) {
    Scalar residual_norm = 0.0f;
    for (int i = 0; i < N; ++i) {
        Scalar err = final_Aug[i * (N + 1) + N] - X_true[i];
        residual_norm += err * err;
    }
    return sqrtf(residual_norm);
}

void gauss_jordan_cpu(Scalar* Aug, int N) {
    for (int k = 0; k < N; ++k) {
        int pivot_row = k;
        Scalar max_val = fabsf(Aug[k * (N + 1) + k]);

        for (int i = k + 1; i < N; ++i) {
            Scalar v = fabsf(Aug[i * (N + 1) + k]);
            if (v > max_val) { max_val = v; pivot_row = i; }
        }

        if (pivot_row != k) {
            for (int j = 0; j < N + 1; ++j)
                swap(Aug[k * (N + 1) + j], Aug[pivot_row * (N + 1) + j]);
        }

        if (fabsf(Aug[k * (N + 1) + k]) < 1.0e-6f) continue;

        Scalar pivot_val = Aug[k * (N + 1) + k];
        for (int j = k; j < N + 1; ++j)
            Aug[k * (N + 1) + j] /= pivot_val;

        for (int i = 0; i < N; ++i) {
            if (i == k) continue;
            Scalar factor = Aug[i * (N + 1) + k];
            for (int j = k; j < N + 1; ++j)
                Aug[i * (N + 1) + j] -= factor * Aug[k * (N + 1) + j];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <N>\n";
        return 1;
    }

    int N = atoi(argv[1]);
    int num_runs = 5;
    double total_time = 0.0;
    Scalar final_residual = 0.0f;

    string xtrue_filename = "xtrue_" + to_string(N) + ".bin";
    Scalar* X_true = read_xtrue(N, xtrue_filename);

    for (int run = 0; run < num_runs; ++run) {
        string matrix_filename = "matrix_" + to_string(N) + ".bin";
        Scalar* Aug = read_matrix(N, matrix_filename);

        clock_t start = clock();
        gauss_jordan_cpu(Aug, N);
        clock_t end = clock();

        total_time += (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
        final_residual = calculate_residual(Aug, X_true, N);

        if (N == 256 && run == 0) {
            cout << "\n--- HASIL VEKTOR SOLUSI X (CPU N=256) ---\n";
            cout.setf(ios::fixed); cout << setprecision(12);
            for (int i = 0; i < N; ++i) {
                if (i < 5 || i >= N - 5)
                    cout << "X[" << i << "] = " << Aug[i * (N + 1) + N] << "\n";
                else if (i == 5) cout << "...\n";
            }
            cout << "Residual Actual: " << final_residual << "\n";
            cout << "---------------------------------------\n";
        }

        delete[] Aug;
    }

    delete[] X_true;
    double avg_time = total_time / num_runs;
    cout.setf(ios::fixed); cout << setprecision(4);
    cout << N << "," << avg_time << "," << final_residual << "\n";
    return 0;
}
