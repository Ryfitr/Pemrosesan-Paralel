#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <cstdlib>
#include <cmath>

using namespace std;

typedef float Scalar;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <N> <ID>\n";
        return 1;
    }

    int N  = atoi(argv[1]);
    int ID = atoi(argv[2]);

    // Vektor solusi sebenarnya X_true
    vector<Scalar> X_true(N);
    for (int i = 0; i < N; ++i) {
        X_true[i] = (Scalar)(i + 1) / (Scalar)N;
    }

    // Matriks augmented [A|b]
    vector<Scalar> Aug((size_t)N * (N + 1));

    // Seed RNG bergantung N dan ID supaya tiap file beda tapi deterministik
    std::mt19937 rng((unsigned int)(1234u + N * 100u + ID));
    std::uniform_real_distribution<float> dist(1.0f, 2.0f);

    // Bangun A acak + buat diagonal dominant
    for (int i = 0; i < N; ++i) {
        Scalar rowsum = 0;
        for (int j = 0; j < N; ++j) {
            Scalar v = dist(rng);
            Aug[(size_t)i * (N + 1) + j] = v;
            rowsum += fabsf(v);
        }
        // Tambah ke diagonal untuk memastikan diagonal dominant
        Aug[(size_t)i * (N + 1) + i] += rowsum;
    }

    // Hitung b = A * X_true dan simpan di kolom terakhir
    for (int i = 0; i < N; ++i) {
        Scalar s = 0;
        for (int j = 0; j < N; ++j) {
            s += Aug[(size_t)i * (N + 1) + j] * X_true[j];
        }
        Aug[(size_t)i * (N + 1) + N] = s;
    }

    // Nama file memakai N dan ID
    string suffix = to_string(N) + "_" + to_string(ID);
    string mfile  = "matrix_" + suffix + ".bin";
    string xfile  = "xtrue_"  + suffix + ".bin";

    ofstream fm(mfile, ios::binary);
    if (!fm.is_open()) {
        cerr << "Error: gagal menulis " << mfile << endl;
        return 1;
    }
    fm.write((char*)Aug.data(), (streamsize)(Aug.size() * sizeof(Scalar)));
    fm.close();

    ofstream fx(xfile, ios::binary);
    if (!fx.is_open()) {
        cerr << "Error: gagal menulis " << xfile << endl;
        return 1;
    }
    fx.write((char*)X_true.data(), (streamsize)(X_true.size() * sizeof(Scalar)));
    fx.close();

    cout << "[GEN-F32] N=" << N << " ID=" << ID
         << " -> " << mfile << " & " << xfile << " OK\n";

    return 0;
}
