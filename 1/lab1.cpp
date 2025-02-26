#include "utils.hpp"

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

    // Используем значение tau, предлагаемое функцией suggestTau
    double tau = suggestTau(N, 0.95);
    cout << "\nИспользуем tau = " << tau << "\n";
    
    const double epsilon = 0.00001;                
    cout << "Точность epsilon = " << epsilon << endl;
    int maxIterations = atoi(argv[2]);

    vector<double> x(N, 0.0);
    double normB = computeNorm(b);
    if (normB == 0.0) {                            
        normB = 1.0;
    }

    auto start = high_resolution_clock::now();
    int iteration = 0;
    while (iteration < maxIterations) {
        vector<double> r = computeResidual(A, x, b);
        double normR = computeNorm(r);

        if (normR / normB < epsilon) {             
            break;
        }

        for (int i = 0; i < N; ++i) {
            x[i] -= tau * r[i];
        }
        ++iteration;
    }

    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    double elapsedSeconds = static_cast<double>(duration.count()) / 1000000.0;

    cout << "\nМетод завершён за " << iteration << " итераций.\n";
    cout << "Время выполнения: " << fixed << setprecision(6) << elapsedSeconds << " секунд.\n";

    vector<double> rFinal = computeResidual(A, x, b);
    double normFinal = computeNorm(rFinal);
    cout << "\n||A*x - b|| = " << fixed << setprecision(6) << normFinal << "\n\n";

    cout << fixed << setprecision(6);
    int nPrint = (N < 50 ? N : 50);
    cout << "Первые " << nPrint << " компонент x:\n";
    for (int i = 0; i < nPrint; ++i) {
        cout << "x[" << i << "] = " << x[i] << "\n";
    }

    return 0;
}
