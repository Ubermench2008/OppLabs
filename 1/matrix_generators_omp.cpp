#include "matrix_generators.h"
#include <cmath>
#include <omp.h>   // Подключаем OpenMP

// Если константа M_PI не определена, определим её.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Функция генерации матрицы A и вектора b для режима 2
void generateMatrix(int N, vector<vector<double>> &A, vector<double> &b) {
    A.assign(N, vector<double>(N, 1.0));
    // Параллельное заполнение диагонали (без директивы schedule)
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        A[i][i] = 2.0;
    }
    b.assign(N, N + 1.0);
}

// Функция генерации матрицы A, модельного решения u и вычисление b = A * u
void generateMatrixModel(int N, vector<vector<double>> &A, vector<double> &b, vector<double> &u) {
    // Генерируем матрицу A
    A.assign(N, vector<double>(N, 1.0));
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        A[i][i] = 2.0;
    }
    // Формируем вектор u
    u.resize(N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        u[i] = sin(2 * M_PI * i / N);
    }
    // Вычисляем b = A * u
    b.assign(N, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += A[i][j] * u[j];
        }
        b[i] = sum;
    }
}

// Функция для подбора tau
double suggestTau(int N, double scale) {
    if (N <= 0) return 0.01; // на всякий случай
    double lamMax = static_cast<double>(N) + 1.0;
    return scale * 2.0 / lamMax;
}
