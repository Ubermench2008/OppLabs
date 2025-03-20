#include "utils/utils.h"
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cblas.h>
#include <iomanip>
#include <random>



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double startTime = 0.0, endTime = 0.0;
    int dims[2] = {0, 0}, periods[2] = {0, 0}, reorder = 0;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc == 3) {
        dims[0] = std::atoi(argv[1]);
        dims[1] = std::atoi(argv[2]);
    } else {
        MPI_Dims_create(size, 2, dims);
    }

    if (rank == 0)
        std::cout << "DIMS: " << dims[0] << " " << dims[1] << std::endl;

    if ((n1 % dims[0] != 0) || (n3 % dims[1] != 0)) {
        if (rank == 0)
            std::cout << "n1, n3 must be divisible by p1, p2" << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Comm gridComm, columnComm, rowComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &gridComm);
    int coords[2];
    MPI_Cart_coords(gridComm, rank, 2, coords);

    int subDims[2];
    subDims[0] = 0; subDims[1] = 1;
    MPI_Cart_sub(gridComm, subDims, &rowComm);
    subDims[0] = 1; subDims[1] = 0;
    MPI_Cart_sub(gridComm, subDims, &columnComm);

    int sub_n = n1 / dims[0];
    int sub_m = n3 / dims[1]; 

    std::vector<double> subA(sub_n * n2);
    std::vector<double> subB(n2 * sub_m);
    std::vector<double> subC(sub_n * sub_m);

    std::vector<double> A, B, C;
    if (coords[0] == 0 && coords[1] == 0) {
        A.resize(n1 * n2);
        B.resize(n2 * n3);
        C.resize(n1 * n3);

        fillMatrixRandom(A, n1 * n2);
        fillMatrixRandom(B, n2 * n3);

        startTime = MPI_Wtime();
    }

    MPI_Datatype SUB_A;
    MPI_Type_contiguous(sub_n * n2, MPI_DOUBLE, &SUB_A);
    MPI_Type_commit(&SUB_A);
    if (coords[1] == 0) {
        MPI_Scatter((coords[0]==0 && coords[1]==0 ? A.data() : nullptr),
                    1, SUB_A,
                    subA.data(), 1, SUB_A, 0, columnComm);
    }
    MPI_Bcast(subA.data(), sub_n * n2, MPI_DOUBLE, 0, rowComm);
    MPI_Type_free(&SUB_A);

    MPI_Datatype SUB_B, SUB_B_RESIZED;
    MPI_Type_vector(n2, sub_m, n3, MPI_DOUBLE, &SUB_B);
    MPI_Type_create_resized(SUB_B, 0, sub_m * sizeof(double), &SUB_B_RESIZED);
    MPI_Type_commit(&SUB_B_RESIZED);
    MPI_Type_free(&SUB_B);

    if (coords[0] == 0) {
        MPI_Scatter(B.data(), 1, SUB_B_RESIZED,
                    subB.data(), n2 * sub_m, MPI_DOUBLE, 0, rowComm);
    }
    MPI_Bcast(subB.data(), n2 * sub_m, MPI_DOUBLE, 0, columnComm);

    for (int row = 0; row < sub_n; row++) {
        for (int col = 0; col < sub_m; col++) {
            subC[row * sub_m + col] = 0.0;
        }
    }
    for (int row = 0; row < sub_n; row++) {
        for (int i = 0; i < n2; i++) {
            double a_val = subA[row * n2 + i];
            for (int col = 0; col < sub_m; col++) {
                subC[row * sub_m + col] += a_val * subB[i * sub_m + col];
            }
        }
    }

    MPI_Datatype SUB_C, SUB_C_RESIZED;
    int sizes[2] = { n1, n3 };
    int subsizes[2] = { sub_n, sub_m };
    int starts[2] = { 0, 0 };
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &SUB_C);
    MPI_Type_create_resized(SUB_C, 0, sub_n * sub_m * sizeof(double), &SUB_C_RESIZED);
    MPI_Type_commit(&SUB_C_RESIZED);
    MPI_Type_free(&SUB_C);

    int localCount = sub_n * sub_m;
    int totalBlocks = dims[0] * dims[1];
    std::vector recvcounts(totalBlocks, 1);
    std::vector displs(totalBlocks, 0);
    for (int proc = 0; proc < totalBlocks; proc++) {
        displs[proc] = proc;
    }

    MPI_Gatherv(subC.data(), localCount, MPI_DOUBLE, (coords[0] == 0 && coords[1] == 0 ? C.data() : nullptr), 
        recvcounts.data(), displs.data(), SUB_C_RESIZED,
        0, gridComm);

    MPI_Type_free(&SUB_C_RESIZED);

    if (rank == 0) {
        endTime = MPI_Wtime();
        std::cout << "Result: " << (endTime - startTime) << " seconds left" << std::endl;
        const double difference = checkResult(A, B, C);
        std::cout << "result difference: " << difference << std::endl;
    }

    MPI_Finalize();
    return 0;
}
