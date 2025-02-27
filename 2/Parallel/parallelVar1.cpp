#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <string>
#include "utils.hpp"

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
    //всем процесса отправляем N
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //создаем матрицу A и b для 0 процесса
    vector<vector<double>> A_full;
    vector<double> b_full;
    if (rank == 0) {
        generateMatrix(N, A_full, b_full);
    }

    /*
Определяем:
rows_per_proc - сколько строк на один процесс без учета остатка
remainder = N % size; - остаток
local_N = rows_per_proc + (rank < remainder ? 1 : 0); - для первых remainder процессов добавляем по 1 строке
(количество строк для каждого процесса)
     */
    int rows_per_proc = N / size;
    int remainder = N % size;
    int local_N = rows_per_proc + (rank < remainder ? 1 : 0);
    vector<int> sendcounts(size), displs(size); //вектор количества строк для процессов i от 1 до size и смещения (по строкам) для процессов i 1..n
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int rows = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = rows;
        displs[i] = offset;
        offset += rows;
    }

    //Одномерная матрица local_N на N эл-тов (для каждого процесса)
    vector<double> A_flat_local(local_N * N);
    if (rank == 0) {
      //Процесс 0 владеет полной одномерной матрицей A
        vector<double> A_flat;
        A_flat.reserve(N * N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A_flat.push_back(A_full[i][j]);
            }
        }
        //делаем вектор распределений ЭЛЕМЕНТОВ по процессам и поэлементных смещений
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
        //Распределение матрицы между процессами:
        //A_flat.data() — указатель на исходный массив
        //sendcounts_flat.data() — массив с количеством элементов для каждого процесса
        //displs_flat.data() — массив смещений
        //MPI_DOUBLE — тип передаваемых данных
        //A_flat_local.data() — куда записывать данные в каждом процессе
        //local_N * N — сколько элементов принимает процесс
        //MPI_DOUBLE — тип принимаемых данных
        //0 — процесс-отправитель
        //MPI_COMM_WORLD — отправляем всем процессам
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                     A_flat_local.data(), local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //остальные процессы просто принимают данные
    }

    //Каждый процесс преобразует свою одномерную матрицу в двумерную
    vector<vector<double>> A_local(local_N, vector<double>(N));
    for (int i = 0; i < local_N; i++) {
        for (int j = 0; j < N; j++) {
            A_local[i][j] = A_flat_local[i * N + j];
        }
    }


    vector<double> b;
    if (rank == 0) {
        b = b_full; //у 0 процесса уже есть b_full
    } else {
        b.resize(N);
    }
    //0 процесс отправляет всем процессам вектор b из N элементов.
    MPI_Bcast(b.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Каждый процесс инициализирует вектор x
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
      //делаем локальный срез по вектору b который есть у каждого процесса
        vector<double> b_local(b.begin() + displs[rank], b.begin() + displs[rank] + local_N);
        vector<double> r_local = computeResidual(A_local, x, b_local); //локальная r

        //считаем локальную норму
        double local_norm_sq = 0.0;
        for (double val : r_local)
            local_norm_sq += val * val;
        double global_norm_sq = 0.0;
        //суммируем local_norm_sq от всех процессов и записываем в global_norm_sq
        MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //&local_norm_sq - адрес данных которые отправляет процесс
        //global_norm_sq - куда запишется результат
        //1 - кол-во передаваемых аргументов от каждого процесса
        //MPI_DOUBLE - тип передаваемых
        //MPI_SUM - операция редукции
        //MPI_COMM_WORLD - по всем процессам
        double global_normR = sqrt(global_norm_sq);

        if (global_normR / normB < 0.00001)
            break;
		//обновление своего x для каждого процесса
        for (int i = 0; i < local_N; i++) {
            int global_i = displs[rank] + i;
            x[global_i] -= tau * r_local[i];
        }

        //собираем обновленный x на по частям со всех процессов
        vector<double> x_updated(N, 0.0);
        MPI_Allgatherv(x.data() + displs[rank], local_N, MPI_DOUBLE,
                       x_updated.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        //x.data() + displs[rank] - локальный блок который передае текущий процесс
        //local_N - сколько отправляется элементов
        //Тип отправляемых
        //x_updated.data() - указатель на массив, в который будет записан результат
        //sendcounts.data() - массив с количеством элементов, получаемых каждым процессом
        //displs.data()	- смещения по каждому процессу
        //Тип принимаемых
        //MPI_COMM_WORLD - группа всех процессов
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
