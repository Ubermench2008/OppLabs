#include "utils/utils.h"

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
    
    if (coords[0] == 0) {
        std::vector<double> B_packed;
        if (coords[1] == 0) {
            B_packed.resize(n2 * n3); // n3 = dims[1] * sub_m
            for (int i = 0; i < dims[1]; i++) {
                for (int row = 0; row < n2; row++) {
                    std::copy(B.begin() + row * n3 + i * sub_m,
                              B.begin() + row * n3 + i * sub_m + sub_m,
                              B_packed.begin() + i * (n2 * sub_m) + row * sub_m);
                }
            }
        }
        MPI_Scatter((coords[1] == 0 ? B_packed.data() : nullptr),
                    n2 * sub_m, MPI_DOUBLE,
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

    int localCount = sub_n * sub_m;
    int totalBlocks = dims[0] * dims[1];
    
    std::vector<int> recvcounts;
    std::vector<int> displs;
    if (coords[0] == 0 && coords[1] == 0) {
        recvcounts.resize(totalBlocks, localCount);
        displs.resize(totalBlocks, 0);

        for (int proc = 0; proc < totalBlocks; proc++) {
            displs[proc] = proc * localCount;
        }
    }
    std::vector<double> tmpC;
    if (coords[0] == 0 && coords[1] == 0) {
        tmpC.resize(totalBlocks * localCount);
    }
    MPI_Gatherv(subC.data(), localCount, MPI_DOUBLE,
                (coords[0] == 0 && coords[1] == 0 ? tmpC.data() : nullptr),
                recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, gridComm);

    if (coords[0] == 0 && coords[1] == 0) {
        for (int proc = 0; proc < totalBlocks; proc++) {
            int proc_row = proc / dims[1]; 
            int proc_col = proc % dims[1]; 
            for (int i = 0; i < sub_n; i++) {
                for (int j = 0; j < sub_m; j++) {
                    int global_row = proc_row * sub_n + i;
                    int global_col = proc_col * sub_m + j;
                    C[global_row * n3 + global_col] = tmpC[proc * localCount + i * sub_m + j];
                }
            }
        }
    }

    if (rank == 0) {
        endTime = MPI_Wtime();
        std::cout << "Result: " << (endTime - startTime) << " seconds left" << std::endl;
        const double difference = checkResult(A, B, C);
        std::cout << "result difference: " << difference << std::endl;
    }

    MPI_Finalize();
    return 0;
}
