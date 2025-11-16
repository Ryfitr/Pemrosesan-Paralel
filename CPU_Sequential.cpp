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

typedef double Scalar;

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

void gauss_jordan_cpu(Scalar* Aug, int N) {
    for (int k = 0; k < N; ++k) {
        
        int pivot_row = k;
        Scalar max_val = std::abs(Aug[k * (N + 1) + k]);

        for (int i = k + 1; i < N; ++i) {
            if (std::abs(Aug[i * (N + 1) + k]) > max_val) {
                max_val = std::abs(Aug[i * (N + 1) + k]);
                pivot_row = i;
            }
        }

        if (pivot_row != k) {
            for (int j = 0; j < N + 1; ++j) {
                Scalar temp = Aug[k * (N + 1) + j];
                Aug[k * (N + 1) + j] = Aug[pivot_row * (N + 1) + j];
                Aug[pivot_row * (N + 1) + j] = temp;
            }
        }

        if (std::abs(Aug[k * (N + 1) + k]) < 1.0e-12) {
            continue; 
        }

        Scalar pivot_val = Aug[k * (N + 1) + k];
        for (int j = k; j < N + 1; ++j) {
            Aug[k * (N + 1) + j] /= pivot_val;
        }

        for (int i = 0; i < N; ++i) {
            if (i != k) {
                Scalar factor = Aug[i * (N + 1) + k];
                for (int j = k; j < N + 1; ++j) {
                    Aug[i * (N + 1) + j] -= factor * Aug[k * (N + 1) + j];
                }
            }
        }
    }
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
        Scalar* Aug = read_matrix(N, matrix_filename);
        
        clock_t start = clock();

        gauss_jordan_cpu(Aug, N);

        clock_t end = clock();
        total_time += (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

        final_residual = calculate_residual(Aug, X_true, N);
        
        if (N == 256 && run == 0) { 
            cout << "\n--- HASIL VEKTOR SOLUSI X (CPU N=256) ---" << endl;
            for (int i = 0; i < N; ++i) {
                if (i < 5 || i >= N - 5) {
                    cout << "X[" << i << "] = " << fixed << setprecision(12) << Aug[i * (N + 1) + N] << endl;
                } else if (i == 5) {
                    cout << "..." << endl; 
                }
            }
            cout << "Residual Actual: " << fixed << setprecision(12) << final_residual << endl;
            cout << "---------------------------------------" << endl;
        }

        delete[] Aug;
    }

    delete[] X_true;
    double avg_time = total_time / num_runs;
    cout << fixed << setprecision(4) << N << "," << avg_time << "," << final_residual << endl;

    return 0;
}