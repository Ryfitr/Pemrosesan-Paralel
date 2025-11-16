#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>

using namespace std;

typedef double Scalar;

void generate_and_save(int N) {
    Scalar* X_true = new Scalar[N];
    for (int i = 0; i < N; ++i) {
        X_true[i] = (Scalar)(i + 1) / (N * 1.0); 
    }
    Scalar* Aug = new Scalar[N * (N + 1)];
    srand(0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Aug[i * (N + 1) + j] = (Scalar)(rand() % 1000 + 1000) / 1000.0;
        }
    }
    for (int i = 0; i < N; ++i) {
        Scalar sum_AX = 0.0;
        for (int j = 0; j < N; ++j) {
            sum_AX += Aug[i * (N + 1) + j] * X_true[j];
        }
        Aug[i * (N + 1) + N] = sum_AX;
    }
    string matrix_filename = "matrix_" + to_string(N) + ".bin";
    ofstream matrix_file(matrix_filename, ios::binary | ios::out);
    matrix_file.write((char*)Aug, N * (N + 1) * sizeof(Scalar));
    matrix_file.close();
    string xtrue_filename = "xtrue_" + to_string(N) + ".bin";
    ofstream xtrue_file(xtrue_filename, ios::binary | ios::out);
    xtrue_file.write((char*)X_true, N * sizeof(Scalar));
    xtrue_file.close();

    cout << "[GENERATOR] N=" << N << " | File " << matrix_filename << " berhasil dibuat." << endl;

    delete[] X_true;
    delete[] Aug;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <matrix_size_N>" << endl;
        return 1;
    }
    int N = atoi(argv[1]);
    generate_and_save(N);
    return 0;
}