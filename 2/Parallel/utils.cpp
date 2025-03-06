#include "utils.hpp"

void generateMatrix(int N, vector<vector<double>> &A, vector<double> &b) {
    A.assign(N, vector<double>(N, 1.0));
    for (int i = 0; i < N; ++i) {
        A[i][i] = 2.0;
    }
    b.assign(N, N + 1.0);
}

double suggestTau(int N, double scale) {
    double lamMax = static_cast<double>(N) + 1.0;
    return scale * 2.0 / lamMax;
}

void printUsage(const char* progName) {
    cerr << "Использование:\n"
         << progName << " 1 # Ручной ввод\n"
         << progName << " 2 # Генерация матрицы/вектора\n";
}

int promptInt(const string &message) {
    int value;
    cout << message;
    cin >> value;
    return value;
}

double promptDouble(const string &message) {
    double value;
    cout << message;
    cin >> value;
    return value;
}

void inputMatrixAndVector(int &N, vector<vector<double>> &A, vector<double> &b) {
    N = promptInt("Введите размер матрицы [N]: ");
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

int getNFromArgsOrDefault(int argc, char* argv[], int defaultN) {
    int N = defaultN;
    if (argc >= 3) {
        N = atoi(argv[2]);
        if (N <= 0) {
            cerr << "Некорректное N, используем N=" << defaultN << " по умолчанию\n";
            N = defaultN;
        }
    } else {
        cerr << "N не задано, используем N=" << defaultN << "\n";
    }
    return N;
}

double computeNorm(const vector<double>& vec) {
    double sum = 0.0;
    for (double x : vec) {
        sum += x * x;
    }
    return sqrt(sum);
}

vector<double> computeResidual(const vector<vector<double>>& A, const vector<double>& x, const vector<double>& b) {
    int local_rows = A.size();
    vector<double> r(local_rows, 0.0);
    for (int i = 0; i < local_rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < x.size(); ++j) {
            sum += A[i][j] * x[j];
        }
        r[i] = sum - b[i];
    }
    return r;
}
