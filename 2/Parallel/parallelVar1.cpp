#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <string>
#include <algorithm>

using namespace std;
using namespace std::chrono;

double computeNorm(const vector<double>& vec) {
    double sum = 0.0;
    for (double x : vec) {
        sum += x * x;
    }
    return sqrt(sum);
}

// Вычисляет r
// b –  вектор правой части, x – вектор решения
vector<double> computeResidual(const vector<double>& A, const vector<double>& x, const vector<double>& b, int N) {
    int local_rows = b.size();
    vector<double> r(local_rows, 0.0);
    for (int i = 0; i < local_rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        r[i] = sum - b[i];
    }
    return r;
}

// Генерирует матрицу A размером NxN и вектор b
// на главное диагонале 2, на остальных позициях 1
// b[i] = N + 1
void generateMatrix(int N, vector<double>& A, vector<double>& b) {
    A.assign(N * N, 1.0);
    for (int i = 0; i < N; i++) {
        A[i * N + i] = 2.0;
    }
    b.assign(N, N + 1.0);
}

// вычисление оптимального тау.

double suggestTau(int N) {
    return 2.0 / (N + 2);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            cerr << "Usage: " << argv[0] << " maxIterations [N]\n";
        MPI_Finalize();
        return 1;
    }

    int maxIterations = atoi(argv[1]);
    int N = (argc >= 3 && atoi(argv[2]) > 0) ? atoi(argv[2]) : 1000;
    if (argc < 3 || N <= 0)
        if (rank == 0)
            cerr << "N not specified or invalid, using N = 1000\n";

    //0 процесс генерирует матрицу A_full и вектор b_full
    vector<double> A_full, b_full;
    if (rank == 0) {
        generateMatrix(N, A_full, b_full);
    }

    //формируем массивы для распределения строк по процессам
    int rows_per_proc = N / size;
    int remainder = N % size;
    int local_N = rows_per_proc + (rank < remainder ? 1 : 0);
    vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int rows = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = rows;
        displs[i] = offset;
        offset += rows;
    }

    // распределяем матрицу по процессам
    // каждый процесс получает local_N строк, local_N * N эл-тов
    vector<double> A_local(local_N * N);
    if (rank == 0) {
        vector<int> sendcounts_flat(size), displs_flat(size);
        offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts_flat[i] = rows * N;
            displs_flat[i] = offset;
            offset += rows * N;
        }
        MPI_Scatterv(A_full.data(), sendcounts_flat.data(), displs_flat.data(), MPI_DOUBLE,
                     A_local.data(), local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                     A_local.data(), local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    //отправляем b по всем процессам
    vector<double> b;
    if (rank == 0) {
        b = b_full;
    } else {
        b.resize(N);
    }
    MPI_Bcast(b.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> x(N, 0.0);

    double normB = computeNorm(b);
    if (normB == 0.0)
        normB = 1.0;
    double tau = suggestTau(N);
    if (rank == 0) {
        cout << "Using tau = " << tau << "\n";
        cout << "Epsilon = 0.00001\n";
    }

    int iteration = 0;
    auto start_time = high_resolution_clock::now();

    while (iteration < maxIterations) {
        //local_N строк вектора b
        vector<double> b_local(b.begin() + displs[rank], b.begin() + displs[rank] + local_N);
        //r_local = A_local * x(full) - b_local
        vector<double> r_local = computeResidual(A_local, x, b_local, N);

        //норма среза, затем редукция по процессам
        double local_norm_sq = 0.0;
        for (double val : r_local)
            local_norm_sq += val * val;
        double global_norm_sq = 0.0;
        MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double global_normR = sqrt(global_norm_sq);

        if (global_normR / normB < 0.00001)
            break;

        //обновляем x
        for (int i = 0; i < local_N; i++) {
            int global_i = displs[rank] + i;
            x[global_i] -= tau * r_local[i];
        }

        //синхронизируем вектор x
        vector<double> x_updated(N, 0.0);
        MPI_Allgatherv(x.data() + displs[rank], local_N, MPI_DOUBLE,
                       x_updated.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        x = x_updated;
        iteration++;
    }

    auto end_time = high_resolution_clock::now();
    double elapsed_seconds = duration_cast<microseconds>(end_time - start_time).count() / 1e6;

    double error = 0.0;
    for (double val : x)
        error = max(error, fabs(1.0 - val));

    if (rank == 0) {
        cout << "Method finished after " << iteration << " iterations.\n";
        cout << "Elapsed time: " << fixed << setprecision(6) << elapsed_seconds << " seconds.\n";
        cout << "max|1 - x[i]| = " << fixed << setprecision(6) << error << "\n";
        int nPrint = (N < 5 ? N : 5);
        cout << "First " << nPrint << " components of x:\n";
        for (int i = 0; i < nPrint; i++) {
            cout << "x[" << i << "] = " << x[i] << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
