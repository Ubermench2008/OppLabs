#include "utils.h"

double checkResult(const std::vector<double>& A, const std::vector<double>& B, const std::vector<double>& C){
	std::vector<double> D(n1 * n3, 0.0);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				n1, n3, n2,
				1.0, A.data(), n2,
					 B.data(), n3,
				0.0, D.data(), n3);

	double maxDiff = 0.0;
	for (size_t i = 0; i < D.size(); ++i) {
		double diff = std::abs(D[i] - C[i]);
		if (diff > maxDiff) {
			maxDiff = diff;
		}
	}

	std::cout << std::fixed << std::setprecision(6);
	std::cout << "Max absolute difference: " << maxDiff << std::endl;

	return maxDiff;
}


void PrintMatrix(const double* matrix, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			std::cout << matrix[i * cols + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void fillMatrixRandom(std::vector<double>& matrix, int countElements) {
  	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1, 1000);

  	for (int i = 0; i < countElements; ++i) {
    	matrix[i] = dist(gen);
  	}
}

