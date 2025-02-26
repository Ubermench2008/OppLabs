#ifndef MATRIX_GENERATORS_H
#define MATRIX_GENERATORS_H

#include <vector>
using std::vector;

/**
 * Генерация матрицы A (N x N):
 *  - диагональ = 2.0,
 *  - вне диагонали = 1.0.
 * Вектор b (длины N) заполняется значениями N+1.
 */
void generateMatrix(int N, vector<vector<double>> &A, vector<double> &b);

/**
 * Генерация матрицы A (N x N) для модельной задачи с произвольным решением:
 *  - диагональ = 2.0,
 *  - вне диагонали = 1.0.
 * Формируется вектор u, элементы которого:
 *      u[i] = sin(2π * i / N)
 * (индексация с i = 0).
 * Затем вычисляется вектор b по правилу: b = A * u.
 * Таким образом, точное решение системы A*x = b равно вектору u.
 */
void generateMatrixModel(int N, vector<vector<double>> &A, vector<double> &b, vector<double> &u);

/**
 * Функция, предлагающая безопасное значение tau по формуле:
 *    tau = scale * 2 / (N + 1)
 * где scale < 1 (например, 0.95), а (N+1) ~ максимальное собственное число матрицы.
 */
double suggestTau(int N, double scale = 0.95);

#endif // MATRIX_GENERATORS_H
