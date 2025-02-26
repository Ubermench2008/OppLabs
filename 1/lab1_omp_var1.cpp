#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <omp.h>  // Подключаем OpenMP
#include "matrix_generators.h"

using namespace std;
using namespace std::chrono;

// Вывод справки
void printUsage(const char* progName) {
    cerr << "Использование:\n"
         << progName << " 1 # Ручной ввод матрицы/вектора\n"
         << progName << " 2 # Генерация матрицы/вектора\n";
}

int promptInt(const string &prompt) {
    int value;
    cout << prompt;
    cin >> value;
    return value;
}

double promptDouble(const string &prompt) {
    double value;
    cout << prompt;
    cin >> value;
    return value;
}

// Ручной ввод
void inputMatrixAndVector(int &N, vector<vector<double>> &A, vector<double> &b) {
    N = promptInt("Введите размер системы (N): ");
    A.assign(N, vector<double>(N, 0.0));
    b.assign(N, 0.0);

    cout << "Введите матрицу A (" << N << "x" << N << "):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cin >> A[i][j];
        }
    }

    cout << "Введите вектор b (" << N << " элементов):\n";
    for (int i = 0; i < N; ++i) {
        cin >> b[i];
    }
}

// Получение N из аргументов командной строки или по умолчанию
int getNFromArgsOrDefault(int argc, char* argv[], int defaultN) {
    int N = defaultN;
    if (argc >= 3) {
        N = atoi(argv[2]);
        if (N <= 0) {
            cerr << "Некорректное N, используем N=" << defaultN << "\n";
            N = defaultN;
        }
    } else {
        cerr << "N не задано, используем N=" << defaultN << "\n";
    }
    return N;
}

// Вычисление евклидовой нормы вектора (распараллеленный цикл)
double computeNorm(const vector<double>& vec) {
    double sum = 0.0;
    // Параллельный цикл без указания schedule
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < (int)vec.size(); ++i) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

// Вычисление невязки r = A*x - b (распараллеленный внешний цикл)
vector<double> computeResidual(const vector<vector<double>>& A,
                               const vector<double>& x,
                               const vector<double>& b)
{
    int N = x.size();
    vector<double> r(N, 0.0);

    // Параллельный цикл без указания schedule
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += A[i][j] * x[j];
        }
        r[i] = sum - b[i];
    }
    return r;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    int mode = atoi(argv[1]);
    int N = 0;
    vector<vector<double>> A;
    vector<double> b;

    if (mode == 1) {
        inputMatrixAndVector(N, A, b);
    } else if (mode == 2) {
        N = getNFromArgsOrDefault(argc, argv, 1000);
        generateMatrix(N, A, b);
        cout << "Сгенерирована матрица " << N << "x" << N
             << "\nСгенерирован вектор b со значениями: " << (N + 1) << "\n";
    } else {
        printUsage(argv[0]);
        return 1;
    }

    double tau = suggestTau(N, 0.95);
    cout << "\nИспользуем tau = " << tau << "\n";

    const double epsilon = 1e-5;
    cout << "Точность epsilon = " << epsilon << endl;
    int maxIterations = atoi(argv[3]);

    vector<double> x(N, 0.0);
    double normB = computeNorm(b);
    if (normB == 0.0) {
        normB = 1.0;
    }

    int iteration = 0;

    auto start = high_resolution_clock::now();
    while (iteration < maxIterations) {
        // Распараллеленное вычисление r = A*x - b (без schedule)
        vector<double> r = computeResidual(A, x, b);
        double normR = computeNorm(r);

        if (normR / normB < epsilon) {
            break;
        }

        // Распараллеленный цикл обновления x (без schedule)
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            x[i] -= tau * r[i];
        }
        ++iteration;
    }
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start);
    double elapsedSeconds = static_cast<double>(duration.count()) / 1e6;

    cout << "\nМетод завершён за " << iteration << " итераций.\n";
    cout << "Время выполнения: " << fixed << setprecision(6) << elapsedSeconds << " секунд.\n";

    vector<double> rFinal = computeResidual(A, x, b);
    double normFinal = computeNorm(rFinal);
    cout << "\n||A*x - b|| = " << fixed << setprecision(6) << normFinal << "\n\n";

    cout << fixed << setprecision(6);
    int nPrint = (N < 10 ? N : 10);
    cout << "Первые " << nPrint << " компонент x:\n";
    for (int i = 0; i < nPrint; ++i) {
        cout << "x[" << i << "] = " << x[i] << "\n";
    }

    return 0;
}
