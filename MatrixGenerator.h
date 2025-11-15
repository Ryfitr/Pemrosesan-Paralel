#ifndef MATRIXGENERATOR_H
#define MATRIXGENERATOR_H

#include <vector>
#include <iostream>
#include <random>
#include <chrono>

using namespace std;

typedef float Scalar; 

void generateSystem(int N, vector<Scalar>& A, vector<Scalar>& B, vector<Scalar>& X_true) {
    mt19937 gen(chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_real_distribution<> distrib(0.0, 10.0);

    A.resize(N * N);
    B.resize(N);
    X_true.resize(N);

    for (int i = 0; i < N; ++i) {
        X_true[i] = distrib(gen);
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = distrib(gen);
        }
    }

    for (int i = 0; i < N; ++i) {
        Scalar sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += A[i * N + j] * X_true[j];
        }
        B[i] = sum;
    }
}

Scalar calculateResidual(int N, const vector<Scalar>& A, const vector<Scalar>& X, const vector<Scalar>& B) {
    Scalar residual_norm_sq = 0.0;
    for (int i = 0; i < N; ++i) {
        Scalar Ax_i = 0.0;
        for (int j = 0; j < N; ++j) {
            Ax_i += A[i * N + j] * X[j];
        }
        Scalar residual_i = Ax_i - B[i];
        residual_norm_sq += residual_i * residual_i;
    }
    return sqrt(residual_norm_sq);
}

#endif