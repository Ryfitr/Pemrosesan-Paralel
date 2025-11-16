#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
using namespace std;

typedef float Scalar;

int main(int argc, char* argv[]) {
    if (argc < 2) { cerr << "Usage: " << argv[0] << " <N>\n"; return 1; }
    int N = atoi(argv[1]);

    vector<Scalar> X_true(N);
    for (int i = 0; i < N; ++i) X_true[i] = (Scalar)(i + 1) / (Scalar)N;

    vector<Scalar> Aug((size_t)N * (N + 1));

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(1.0f, 2.0f);

    for (int i = 0; i < N; ++i) {
        Scalar rowsum = 0;
        for (int j = 0; j < N; ++j) {
            Scalar v = dist(rng);
            Aug[(size_t)i * (N + 1) + j] = v;
            rowsum += fabsf(v);
        }
        Aug[(size_t)i * (N + 1) + i] += rowsum;
    }

    for (int i = 0; i < N; ++i) {
        Scalar s = 0;
        for (int j = 0; j < N; ++j) s += Aug[(size_t)i * (N + 1) + j] * X_true[j];
        Aug[(size_t)i * (N + 1) + N] = s;
    }

    string mfile = "matrix_" + to_string(N) + ".bin";
    string xfile = "xtrue_"  + to_string(N) + ".bin";

    ofstream fm(mfile, ios::binary); fm.write((char*)Aug.data(), (streamsize)(Aug.size()*sizeof(Scalar))); fm.close();
    ofstream fx(xfile, ios::binary); fx.write((char*)X_true.data(), (streamsize)(X_true.size()*sizeof(Scalar))); fx.close();

    cout << "[GEN-F32] N=" << N << " -> " << mfile << " & " << xfile << " OK\n";
    return 0;
}
