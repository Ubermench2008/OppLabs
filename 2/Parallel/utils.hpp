#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <string>

using namespace std;
using namespace std::chrono;


void generateMatrix(int N, vector<vector<double>> &A, vector<double> &b);
double suggestTau(int N, double scale);
void printUsage(const char* progName);
int promptInt(const string &message);
double promptDouble(const string &message);
void inputMatrixAndVector(int &N, vector<vector<double>> &A, vector<double> &b);
int getNFromArgsOrDefault(int argc, char* argv[], int defaultN);
double computeNorm(const vector<double>& vec);
vector<double> computeResidual(const vector<vector<double>>& A, const vector<double>& x, const vector<double>& b);

