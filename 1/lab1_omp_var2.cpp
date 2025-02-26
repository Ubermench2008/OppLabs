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

// Функция для вывода подсказки по использованию программы
void printUsage(const char* progName) {
    cerr << "Использование:\n"
         << progName << " 1 # Ручной ввод матрицы/вектора\n"
         << progName << " 2 [N] # Генерация матрицы/вектора (diag=2, offdiag=1, b[i]=N+1)\n";
}

// Запрос целого числа
int promptInt(const string &prompt) {
    int value;
    cout << prompt;
    cin >> value;
    return value;
}

// Ручной ввод матрицы и вектора (режим 1)
void inputMatrixAndVector(int &N, vector<vector<double>> &A, vector<double> &b) {
    N = promptInt("Введите размер системы (N): ");
    A.assign(N, vector<double>(N, 0.0));
    b.assign(N, 0.0);

    cout << "Введите матрицу A (" << N << "x" << N << "):\n";
    for (int i = 0; i < N; ++i) {
        cout << "Строка " << i + 1 << " (введите " << N << " чисел): ";
        for (int j = 0; j < N; ++j) {
            cin >> A[i][j];
        }
    }

    cout << "Введите вектор b (" << N << " элементов):\n";
    for (int i = 0; i < N; ++i) {
        cout << "b[" << i << "]: ";
        cin >> b[i];
    }
}

// Получение N из аргументов командной строки или значение по умолчанию
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

// Последовательное вычисление нормы вектора (используется для вывода)
double computeNormSeq(const vector<double>& vec) {
    double sum = 0.0;
    for (double x : vec) {
        sum += x * x;
    }
    return sqrt(sum);
}

// Последовательное вычисление невязки (используется для итогового вывода)
vector<double> computeResidualSeq(const vector<vector<double>>& A,
                                  const vector<double>& x,
                                  const vector<double>& b)
{
    int N = x.size();
    vector<double> r(N, 0.0);
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

    // Выбор режима
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

    // Параметр tau
    double tau = suggestTau(N, 0.95);
    cout << "\nИспользуем tau = " << tau << "\n";

    const double epsilon = 1e-5;
    cout << "Точность epsilon = " << epsilon << endl;
    int maxIterations = atoi(argv[3]);

    // Инициализация вектора решения
    vector<double> x(N, 0.0);

    // Норма b (для критерия остановки)
    double normB = computeNormSeq(b);
    if (normB == 0.0) {
        normB = 1.0;
    }

    auto startTime = high_resolution_clock::now();

    int iteration = 0;
    double localSum;
    bool needStop = false;
    
    // Один параллельный регион на весь алгоритм
    #pragma omp parallel shared(needStop, x, A, b, iteration)
    {
        // Используем цикл for по итерациям вместо while(true)
        for (int it = 0; it < maxIterations; it++) {
            // Вычисляем невязку r = A*x - b (параллельно)
            vector<double> r(N, 0.0);
            #pragma omp for
            for (int i = 0; i < N; ++i) {
                double sum = 0.0;
                for (int j = 0; j < N; ++j) {
                    sum += A[i][j] * x[j];
                }
                r[i] = sum - b[i];
            }

            double normR;
            localSum = 0.0;
            
            #pragma omp for reduction(+:localSum)
            for (int i = 0; i < N; ++i) {
                localSum += r[i] * r[i];
            }
            normR = sqrt(localSum);

            // Один поток проверяет критерий остановки
            #pragma omp single
            {
                iteration = it + 1; // Итерации считаем с 1
                if ((normR / normB) < epsilon) {
                    needStop = true;
                }
            }

            #pragma omp barrier

            if (needStop) {
                break;
            }

            // Обновляем x параллельно: x = x - tau * r
            #pragma omp for
            for (int i = 0; i < N; ++i) {
                x[i] -= tau * r[i];
            }

            #pragma omp barrier
        } // for(it)
    } // конец параллельного региона

    auto endTime = high_resolution_clock::now();
    double elapsedSeconds = duration_cast<microseconds>(endTime - startTime).count() / 1e6;

    cout << "\nМетод завершён за " << iteration << " итераций.\n";
    cout << "Время выполнения: " << fixed << setprecision(6)
         << elapsedSeconds << " секунд.\n";

    // Последовательное вычисление финальной невязки (только для вывода)
    vector<double> rFinal = computeResidualSeq(A, x, b);
    double normFinal = computeNormSeq(rFinal);
    cout << "\n||A*x - b|| = " << fixed << setprecision(6) << normFinal << "\n\n";

    // Вывод первых нескольких компонент решения
    int nPrint = (N < 10 ? N : 10);
    cout << fixed << setprecision(6);
    cout << "Первые " << nPrint << " компонент x:\n";
    for (int i = 0; i < nPrint; ++i) {
        cout << "x[" << i << "] = " << x[i] << "\n";
    }

    return 0;
}
