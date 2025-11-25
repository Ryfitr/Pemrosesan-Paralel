#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <string>

using namespace std;

typedef float Scalar;

static inline Scalar* read_matrix(int N, const string& filename) {
    Scalar* Aug = new Scalar[(size_t)N * (N + 1)];
    ifstream file(filename, ios::binary | ios::in);
    if (!file.is_open()) {
        cerr << "Error: Tidak dapat membaca " << filename << endl;
        exit(1);
    }
    file.read((char*)Aug, (streamsize)((size_t)N * (N + 1) * sizeof(Scalar)));
    file.close();
    return Aug;
}

static inline Scalar* read_xtrue(int N, const string& filename) {
    Scalar* X_true = new Scalar[N];
    ifstream file(filename, ios::binary | ios::in);
    if (!file.is_open()) {
        cerr << "Error: Tidak dapat membaca " << filename << endl;
        exit(1);
    }
    file.read((char*)X_true, (streamsize)((size_t)N * sizeof(Scalar)));
    file.close();
    return X_true;
}

// Menghitung norm error solusi: ||x_computed - X_true||_2
static inline Scalar calculate_residual(Scalar* final_Aug, Scalar* X_true, int N) {
    Scalar residual_norm = 0.0f;
    for (int i = 0; i < N; ++i) {
        Scalar err = final_Aug[(size_t)i * (N + 1) + N] - X_true[i];
        residual_norm += err * err;
    }
    return sqrtf(residual_norm);
}

void gauss_jordan_cpu(Scalar* Aug, int N) {
    for (int k = 0; k < N; ++k) {
        // Cari pivot (partial pivoting)
        int pivot_row = k;
        Scalar max_val = fabsf(Aug[(size_t)k * (N + 1) + k]);

        for (int i = k + 1; i < N; ++i) {
            Scalar v = fabsf(Aug[(size_t)i * (N + 1) + k]);
            if (v > max_val) {
                max_val = v;
                pivot_row = i;
            }
        }

        // Tukar baris jika perlu
        if (pivot_row != k) {
            for (int j = 0; j < N + 1; ++j) {
                swap(Aug[(size_t)k * (N + 1) + j],
                     Aug[(size_t)pivot_row * (N + 1) + j]);
            }
        }

        // Jika pivot terlalu kecil, skip (matriks hampir singular)
        if (fabsf(Aug[(size_t)k * (N + 1) + k]) < 1.0e-6f) continue;

        // Normalisasi pivot row
        Scalar pivot_val = Aug[(size_t)k * (N + 1) + k];
        for (int j = k; j < N + 1; ++j) {
            Aug[(size_t)k * (N + 1) + j] /= pivot_val;
        }

        // Eliminasi baris lain
        for (int i = 0; i < N; ++i) {
            if (i == k) continue;
            Scalar factor = Aug[(size_t)i * (N + 1) + k];
            for (int j = k; j < N + 1; ++j) {
                Aug[(size_t)i * (N + 1) + j] -= factor * Aug[(size_t)k * (N + 1) + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <N> <ID>\n";
        return 1;
    }

    int N  = atoi(argv[1]);
    int ID = atoi(argv[2]);

    string suffix          = to_string(N) + "_" + to_string(ID);
    string xtrue_filename  = "xtrue_"  + suffix + ".bin";
    string matrix_filename = "matrix_" + suffix + ".bin";

    Scalar* X_true = read_xtrue(N, xtrue_filename);
    Scalar* Aug    = read_matrix(N, matrix_filename);

    clock_t start = clock();
    gauss_jordan_cpu(Aug, N);
    clock_t end   = clock();

    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    Scalar final_residual = calculate_residual(Aug, X_true, N);

    // Output detail solusi ke terminal (hanya N=256, ID=1)
    if (N == 256 && ID == 1) {
        cout << "\n--- HASIL VEKTOR SOLUSI X (CPU N=" << N << ", ID=" << ID << ") ---\n";
        cout.setf(ios::fixed);
        cout << setprecision(12);
        for (int i = 0; i < N; ++i) {
            if (i < 5 || i >= N - 5) {
                cout << "X[" << i << "] = " << Aug[(size_t)i * (N + 1) + N] << "\n";
            } else if (i == 5) {
                cout << "...\n";
            }
        }
        cout << "Residual Actual: " << final_residual << "\n";
        cout << "---------------------------------------\n";
    }

    delete[] Aug;
    delete[] X_true;

    // Cetak baris data ke terminal
    cout.setf(ios::fixed);
    cout << setprecision(4);
    cout << N << "," << ID << "," << ms << "," << final_residual << "\n";

    // Tulis juga ke file CSV
    string csv_name = "cpu_results.csv";
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
        csv << N << "," << ID << "," << ms << "," << final_residual << "\n";
        csv.close();
    } else {
        cerr << "Peringatan: tidak bisa membuka " << csv_name << " untuk menulis.\n";
    }

    return 0;
}
