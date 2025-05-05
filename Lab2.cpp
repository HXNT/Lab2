#include <iostream>
#include <complex>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <cblas.h>
#include <cmath>
#include <new>
#include <iomanip>

#define N 4096
#define NAIVE_MAX 1024
#define BLOCK_SIZE 64

using namespace std;
using namespace chrono;

typedef complex<float> Complex;

void generate_matrix(Complex* A, int size = N) {
    for (int i = 0; i < size * size; ++i) {
        A[i] = Complex((float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
    }
}

void zero_matrix(Complex* A, int size = N) {
    memset(A, 0, sizeof(Complex) * size * size);
}

// 1. Naive Method
void multiply_naive(const Complex* A, const Complex* B, Complex* C) {
    const int size = NAIVE_MAX;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            Complex sum = 0;
            for (int k = 0; k < size; ++k) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

// 2. BLAS
void multiply_blas(const Complex* A, const Complex* B, Complex* C) {
    const Complex alpha(1.0f, 0.0f);
    const Complex beta(0.0f, 0.0f);
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        N, N, N, &alpha, A, N, B, N, &beta, C, N);
}

// 3. Block Method
void multiply_blocked(const Complex* A, const Complex* B, Complex* C) {
    zero_matrix(C);
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                int i_end = min(ii + BLOCK_SIZE, N);
                int j_end = min(jj + BLOCK_SIZE, N);
                int k_end = min(kk + BLOCK_SIZE, N);

                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; ++j) {
                        Complex sum = C[i * N + j];
                        for (int k = kk; k < k_end; ++k) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}


bool compare_matrices(const Complex* A, const Complex* B, int size, float epsilon = 1e-3f) {
    for (int i = 0; i < size * size; ++i) {
        if (abs(A[i] - B[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main() {
    try {
        cout << "Matrix Multiplication Benchmark (N=" << N << ")" << endl;
        cout << "======================================" << endl;

        Complex* A = new Complex[N * N];
        Complex* B = new Complex[N * N];
        Complex* C_blas = new Complex[N * N];
        Complex* C_blocked = new Complex[N * N];

        Complex* A_naive = new Complex[NAIVE_MAX * NAIVE_MAX];
        Complex* B_naive = new Complex[NAIVE_MAX * NAIVE_MAX];
        Complex* C_naive = new Complex[NAIVE_MAX * NAIVE_MAX];

        cout << "Generating matrices..." << endl;
        generate_matrix(A);
        generate_matrix(B);
        generate_matrix(A_naive, NAIVE_MAX);
        generate_matrix(B_naive, NAIVE_MAX);

        struct BenchmarkResult {
            string name;
            double time;
            double mflops;
            int size;
        };

        BenchmarkResult* results = new BenchmarkResult[3];
        int resultIndex = 0;

        // 1. Test Naive Method
        {
            cout << "\n1. Testing naive multiplication (N=" << NAIVE_MAX << ")..." << endl;
            zero_matrix(C_naive, NAIVE_MAX);
            auto start = high_resolution_clock::now();
            multiply_naive(A_naive, B_naive, C_naive);
            auto end = high_resolution_clock::now();
            double duration = chrono::duration_cast<chrono::duration<double>>(end - start).count();
            double mflops = (2.0 * pow(NAIVE_MAX,3.0)) / (duration * pow(10.0,-6.0));

            results[resultIndex++] = { "Naive", duration, mflops, NAIVE_MAX };

            cout << fixed << setprecision(3);
            cout << "Time: " << duration << " sec\n";
            cout << "Performance: " << mflops << " MFLOPS\n";
        }

        // 2. Test BLAS
        {
            cout << "\n2. Testing BLAS cgemm (N=" << N << ")..." << endl;
            zero_matrix(C_blas);
            auto start = high_resolution_clock::now();
            multiply_blas(A, B, C_blas);
            auto end = high_resolution_clock::now();
            double duration = chrono::duration_cast<chrono::duration<double>>(end - start).count();
            double mflops = (2.0 * pow(N,3.0)) / (duration * pow(10.0,-6.0));

            results[resultIndex++] = { "BLAS", duration, mflops, N };

            cout << fixed << setprecision(3);
            cout << "Time: " << duration << " sec\n";
            cout << "Performance: " << mflops << " MFLOPS\n";
        }

        // 3. Test Block Method
        {
            cout << "\n3. Testing blocked multiplication (N=" << N << ")..." << endl;
            zero_matrix(C_blocked);
            auto start = high_resolution_clock::now();
            multiply_blocked(A, B, C_blocked);
            auto end = high_resolution_clock::now();
            double duration = chrono::duration_cast<chrono::duration<double>>(end - start).count();
            double mflops = (2.0 * pow(N,3.0)) / (duration * pow(10.0,-6.0));

            results[resultIndex++] = { "Blocked", duration, mflops, N };

            cout << fixed << setprecision(3);
            cout << "Time: " << duration << " sec\n";
            cout << "Performance: " << mflops << " MFLOPS\n";
        }

        // Comparison BLAS and Block
        cout << "\nComparing BLAS and Blocked results..." << endl;
        if (compare_matrices(C_blas, C_blocked, N)) {
            cout << "Results match within tolerance\n";
        }
        else {
            cout << "WARNING: Results differ!\n";
        }

        cout << "\nPerformance Comparison Summary:\n";
        cout << "================================================================\n";
        cout << "| Method    | Matrix Size | Time (sec) | MFLOPS     | Speedup   |\n";
        cout << "================================================================\n";

        for (int i = 0; i < resultIndex; ++i) {
            double speedup = (i > 0) ? results[0].time / results[i].time *
                (results[i].size * results[i].size * results[i].size) /
                (results[0].size * results[0].size * results[0].size) : 1.0;

            cout << "| " << setw(9) << left << results[i].name << " | "
                << setw(11) << results[i].size << " | "
                << setw(9) << setprecision(3) << results[i].time << " | "
                << setw(10) << setprecision(1) << scientific << results[i].mflops << " | "
                << setw(9) << fixed << setprecision(1) << speedup << "x |\n";
        }
        cout << "================================================================\n";

        cout << "This work was made by student Kotsko Oleg Evgenievich 090304RPIb-o24" << endl;

        delete[] A;
        delete[] B;
        delete[] C_blas;
        delete[] C_blocked;
        delete[] A_naive;
        delete[] B_naive;
        delete[] C_naive;
        delete[] results;

    }
    catch (const bad_alloc& e) {
        cerr << "Memory allocation error: " << e.what() << endl;
        return 1;
    }
    catch (...) {
        cerr << "Unknown error occurred!" << endl;
        return 1;
    }

    return 0;
}