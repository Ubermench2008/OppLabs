#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <string>
#include "utils.hpp"  // Функции generateMatrix, suggestTau, computeNorm и computeResidual определены в utils.hpp

using namespace std;
using namespace std::chrono;

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

    int N = 0;
    if (rank == 0) {
        N = (argc >= 3 && atoi(argv[2]) > 0) ? atoi(argv[2]) : 1000;
        if (argc < 3 || N <= 0)
            cerr << "N not specified or invalid, using N = 1000\n";
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<vector<double>> A_full;
    vector<double> b_full;
    if (rank == 0) {
        generateMatrix(N, A_full, b_full);
    }

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

    vector<double> A_flat_local(local_N * N);
    if (rank == 0) {
        vector<double> A_flat;
        A_flat.reserve(N * N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A_flat.push_back(A_full[i][j]);
            }
        }
        vector<int> sendcounts_flat(size), displs_flat(size);
        offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts_flat[i] = rows * N;
            displs_flat[i] = offset;
            offset += rows * N;
        }
        MPI_Scatterv(A_flat.data(), sendcounts_flat.data(), displs_flat.data(), MPI_DOUBLE,
                     A_flat_local.data(), local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                     A_flat_local.data(), local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    vector<vector<double>> A_local(local_N, vector<double>(N));
    for (int i = 0; i < local_N; i++) {
        for (int j = 0; j < N; j++) {
            A_local[i][j] = A_flat_local[i * N + j];
        }
    }

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
    double tau = suggestTau(N, 0.95);
    if (rank == 0) {
        cout << "Using tau = " << tau << "\n";
        cout << "Epsilon = 0.00001\n";
    }

    int iteration = 0;
    auto start_time = high_resolution_clock::now();

    while (iteration < maxIterations) {
        vector<double> b_local(b.begin() + displs[rank], b.begin() + displs[rank] + local_N);
        vector<double> r_local = computeResidual(A_local, x, b_local);

        double local_norm_sq = 0.0;
        for (double val : r_local)
            local_norm_sq += val * val;
        double global_norm_sq = 0.0;
        MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double global_normR = sqrt(global_norm_sq);

        if (global_normR / normB < 0.00001)
            break;

        for (int i = 0; i < local_N; i++) {
            int global_i = displs[rank] + i;
            x[global_i] -= tau * r_local[i];
        }

        vector<double> x_updated(N, 0.0);
        MPI_Allgatherv(x.data() + displs[rank], local_N, MPI_DOUBLE,
                       x_updated.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        x = x_updated;
        iteration++;
    }

    auto end_time = high_resolution_clock::now();
    double elapsed_seconds = duration_cast<microseconds>(end_time - start_time).count() / 1e6;

    vector<double> b_local(b.begin() + displs[rank], b.begin() + displs[rank] + local_N);
    vector<double> r_local = computeResidual(A_local, x, b_local);
    double local_norm_sq = 0.0;
    for (double val : r_local)
        local_norm_sq += val * val;
    double final_norm_sq = 0.0;
    MPI_Allreduce(&local_norm_sq, &final_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double final_norm = sqrt(final_norm_sq);

    if (rank == 0) {
        cout << "Method finished after " << iteration << " iterations.\n";
        cout << "Elapsed time: " << fixed << setprecision(6) << elapsed_seconds << " seconds.\n";
        cout << "||A*x - b|| = " << fixed << setprecision(6) << final_norm << "\n";
        int nPrint = (N < 5 ? N : 5);
        cout << "First " << nPrint << " components of x:\n";
        for (int i = 0; i < nPrint; i++) {
            cout << "x[" << i << "] = " << x[i] << "\n";
        }
    }
    MPI_Finalize();
    return 0;
}
