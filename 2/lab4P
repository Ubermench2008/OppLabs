#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <iomanip>
#include <algorithm>

constexpr double eps = 1e-9;
constexpr double a = 1e5;

constexpr double X0 = -1;
constexpr double Y0 = -1;
constexpr double Z0 = -1;

constexpr double Dx = 2.0;
constexpr double Dy = 2.0;
constexpr double Dz = 2.0;

constexpr int Nx = 320;
constexpr int Ny = 320;
constexpr int Nz = 320;

constexpr double hx = Dx / (Nx - 1.0);
constexpr double hy = Dy / (Ny - 1.0);
constexpr double hz = Dz / (Nz - 1.0);

double phi_function(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double rho(double x, double y, double z) {
    return 6 - a * phi_function(x, y, z);
}

inline int index3D(int x, int y, int z) {
    return z * (Nx * Ny) + y * Nx + x;
}

double JacobiMethod(int rank, int layerHeight, const std::vector<double>& prevPhi, int x, int y, int z) {
    double xComp = (prevPhi[index3D(x - 1, y, z)] + prevPhi[index3D(x + 1, y, z)]) / (hx * hx);
    double yComp = (prevPhi[index3D(x, y - 1, z)] + prevPhi[index3D(x, y + 1, z)]) / (hy * hy);
    double zComp = (prevPhi[index3D(x, y, z - 1)] + prevPhi[index3D(x, y, z + 1)]) / (hz * hz);
    double mult = 1.0 / (2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + a);
    double x_coord = X0 + x * hx;
    double y_coord = Y0 + y * hy;
    double z_coord = Z0 + (z + layerHeight * rank) * hz;
    return mult * (xComp + yComp + zComp - rho(x_coord, y_coord, z_coord));
}

double JacobiMethodTopEdge(int rank, int layerHeight, const std::vector<double>& prevPhi, int x, int y, const std::vector<double>& upLayer) {
    int z = layerHeight - 1;
    double xComp = (prevPhi[index3D(x - 1, y, z)] + prevPhi[index3D(x + 1, y, z)]) / (hx * hx);
    double yComp = (prevPhi[index3D(x, y - 1, z)] + prevPhi[index3D(x, y + 1, z)]) / (hy * hy);
    double zComp = (prevPhi[index3D(x, y, z - 1)] + upLayer[y * Nx + x]) / (hz * hz);
    double mult = 1.0 / (2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + a);
    double x_coord = X0 + x * hx;
    double y_coord = Y0 + y * hy;
    double z_coord = Z0 + ((layerHeight - 1) + layerHeight * rank) * hz;
    return mult * (xComp + yComp + zComp - rho(x_coord, y_coord, z_coord));
}

double JacobiMethodBottomEdge(int rank, int layerHeight, const std::vector<double>& prevPhi, int x, int y, const std::vector<double>& downLayer) {
    int z = 0;
    double xComp = (prevPhi[index3D(x - 1, y, z)] + prevPhi[index3D(x + 1, y, z)]) / (hx * hx);
    double yComp = (prevPhi[index3D(x, y - 1, z)] + prevPhi[index3D(x, y + 1, z)]) / (hy * hy);
    double zComp = (downLayer[y * Nx + x] + prevPhi[index3D(x, y, z + 1)]) / (hz * hz);
    double mult = 1.0 / (2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + a);
    double x_coord = X0 + x * hx;
    double y_coord = Y0 + y * hy;
    double z_coord = Z0 + (layerHeight * rank) * hz;
    return mult * (xComp + yComp + zComp - rho(x_coord, y_coord, z_coord));
}

void CalculateCenter(int layerHeight, const std::vector<double>& prevPhi, std::vector<double>& Phi, int rank, char& flag) {
    for (int z = 1; z < layerHeight - 1; ++z) {
        for (int y = 1; y < Ny - 1; ++y) {
            for (int x = 1; x < Nx - 1; ++x) {
                int idx = index3D(x, y, z);
                Phi[idx] = JacobiMethod(rank, layerHeight, prevPhi, x, y, z);
                if (std::fabs(Phi[idx] - prevPhi[idx]) > eps) {
                    flag = 0;
                }
            }
        }
    }
}

void CalculateEdges(int layerHeight, const std::vector<double>& prevPhi, std::vector<double>& Phi, int rank, char& flag,
                    const std::vector<double>& downLayer, const std::vector<double>& upLayer, int size) {
    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 1; x < Nx - 1; ++x) {
            if (rank != 0) {
                int idx = index3D(x, y, 0);
                Phi[idx] = JacobiMethodBottomEdge(rank, layerHeight, prevPhi, x, y, downLayer);
                if (std::fabs(Phi[idx] - prevPhi[idx]) > eps) {
                    flag = 0;
                }
            }
            if (rank != size - 1) {
                int idx = index3D(x, y, layerHeight - 1);
                Phi[idx] = JacobiMethodTopEdge(rank, layerHeight, prevPhi, x, y, upLayer);
                if (std::fabs(Phi[idx] - prevPhi[idx]) > eps) {
                    flag = 0;
                }
            }
        }
    }
}

void CalculateMaxDifference(int rank, int layerHeight, const std::vector<double>& Phi) {
    double maxDiff = 0.0;
    for (int z = 0; z < layerHeight; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                double x_coord = X0 + x * hx;
                double y_coord = Y0 + y * hy;
                double z_coord = Z0 + (z + layerHeight * rank) * hz;
                double diff = std::fabs(Phi[index3D(x, y, z)] - phi_function(x_coord, y_coord, z_coord));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }
    }
    double globalMax = 0.0;
    MPI_Allreduce(&maxDiff, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Max difference: " << globalMax << std::endl;
    }
}

int main(int argc, char** argv) {
    int rank, size;
    double timeStart = 0.0, timeFinish = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int layerHeight = Nz / size;
    std::vector<double> Phi(Nx * Ny * layerHeight, 0.0);
    std::vector<double> prevPhi(Nx * Ny * layerHeight, 0.0);
    std::vector<double> downLayer(Nx * Ny, 0.0);
    std::vector<double> upLayer(Nx * Ny, 0.0);

    if (rank == 0) {
        timeStart = MPI_Wtime();
    }

    for (int z = 0; z < layerHeight; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int idx = index3D(x, y, z);
                double x_coord = X0 + x * hx;
                double y_coord = Y0 + y * hy;
                double z_coord = Z0 + (z + layerHeight * rank) * hz;
                if (y == 0 || x == 0 || y == Ny - 1 || x == Nx - 1) {
                    Phi[idx] = phi_function(x_coord, y_coord, z_coord);
                    prevPhi[idx] = phi_function(x_coord, y_coord, z_coord);
                } else {
                    Phi[idx] = 0.0;
                    prevPhi[idx] = 0.0;
                }
            }
        }
    }

    if (rank == 0) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int idx = index3D(x, y, 0);
                double x_coord = X0 + x * hx;
                double y_coord = Y0 + y * hy;
                double z_coord = Z0;
                Phi[idx] = phi_function(x_coord, y_coord, z_coord);
                prevPhi[idx] = phi_function(x_coord, y_coord, z_coord);
            }
        }
    }

    if (rank == size - 1) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int idx = index3D(x, y, layerHeight - 1);
                double x_coord = X0 + x * hx;
                double y_coord = Y0 + y * hy;
                double z_coord = Z0 + Dz;
                Phi[idx] = phi_function(x_coord, y_coord, z_coord);
                prevPhi[idx] = phi_function(x_coord, y_coord, z_coord);
            }
        }
    }

    int counter = 0;
    MPI_Request requests[4];

    char isDiverged = 0;
    while (!isDiverged) {
        isDiverged = 1;
        std::swap(prevPhi, Phi);

        if (rank != 0) {
            MPI_Isend(prevPhi.data(), Nx * Ny, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(downLayer.data(), Nx * Ny, MPI_DOUBLE, rank - 1, 20, MPI_COMM_WORLD, &requests[1]);
        }
        if (rank != size - 1) {
            MPI_Isend(prevPhi.data() + (layerHeight - 1) * Nx * Ny, Nx * Ny, MPI_DOUBLE, rank + 1, 20, MPI_COMM_WORLD, &requests[2]);
            MPI_Irecv(upLayer.data(), Nx * Ny, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD, &requests[3]);
        }

        CalculateCenter(layerHeight, prevPhi, Phi, rank, isDiverged);

        if (rank != 0) {
            MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
        }
        if (rank != size - 1) {
            MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
        }

        CalculateEdges(layerHeight, prevPhi, Phi, rank, isDiverged, downLayer, upLayer, size);

        char globalFlag;
        MPI_Allreduce(&isDiverged, &globalFlag, 1, MPI_CHAR, MPI_LAND, MPI_COMM_WORLD);
        isDiverged = globalFlag;
        counter++;
    }

    if (rank == 0) {
        timeFinish = MPI_Wtime();
    }

    int maxCounter;
    MPI_Allreduce(&counter, &maxCounter, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    counter = maxCounter;

    CalculateMaxDifference(rank, layerHeight, Phi);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Number of iterations: " << counter << std::endl;
        std::cout << "Time: " << (timeFinish - timeStart) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
