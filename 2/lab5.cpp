#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <unistd.h>

#define N_TASKS 320
#define REQUEST_TAG 0
#define TASK_TAG 1

struct Task {
    int taskNumber;
    int difficulty;
    bool completed;
};

std::mutex mtx;
std::vector<Task> task_list;

void calculate_task(Task &task, int rank) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!task.completed) {
            task.completed = true;
        } else {
            return;
        }
    }
    usleep(task.difficulty);
}

int request_task(int request_rank) {
    int task_id = -1;
    int request = 1;
    MPI_Send(&request, 1, MPI_INT, request_rank, REQUEST_TAG, MPI_COMM_WORLD);
    MPI_Recv(&task_id, 1, MPI_INT, request_rank, TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return task_id;
}

void worker_thread(int rank, int size) {
    while (true) {
        Task* task_to_do = nullptr;
        int local_start = rank * N_TASKS / size;
        int local_end = local_start + N_TASKS / size;
        for (int i = local_start; i < local_end; ++i) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (!task_list[i].completed) {
                    task_to_do = &task_list[i];
                    break;
                }
            }
        }

        if (task_to_do == nullptr) {
            for (int i = 0; i < size; ++i) {
                if (i != rank) {
                    int task_id = request_task(i);
                    if (task_id != -1) {
                        task_to_do = &task_list[task_id];
                        break;
                    }
                }
            }
            if (task_to_do == nullptr) {
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Send(nullptr, 0, MPI_INT, rank, REQUEST_TAG, MPI_COMM_WORLD);
                break;
            }
        }
        calculate_task(*task_to_do, rank);
    }
}

void server_thread(int rank, int size) {
    int request;
    MPI_Status status;
    while (true) {
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_SOURCE == rank) {
            break;
        }
        int task_id = -1;
        int local_start = rank * N_TASKS / size;
        int local_end = local_start + N_TASKS / size;
        for (int i = local_start; i < local_end; ++i) {
            std::lock_guard<std::mutex> lock(mtx);
            if (!task_list[i].completed) {
                task_id = i;
                task_list[i].completed = true;
                break;
            }
        }
        MPI_Send(&task_id, 1, MPI_INT, status.MPI_SOURCE, TASK_TAG, MPI_COMM_WORLD);
    }
}

int main(int argc, char* argv[]) {
    int rank, size, provided;
    double start_time, end_time;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        std::cerr << "Can't init thread\n";
        MPI_Finalize();
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    task_list.resize(N_TASKS);
    if (rank == 0) {
        for (int i = 0; i < N_TASKS; ++i) {
            task_list[i].taskNumber = i;
            task_list[i].difficulty = (i + 1) * 1000;
            task_list[i].completed = false;
        }
    }
    MPI_Bcast(task_list.data(), N_TASKS * sizeof(Task), MPI_BYTE, 0, MPI_COMM_WORLD);

    std::thread worker(worker_thread, rank, size);
    std::thread server(server_thread, rank, size);

    start_time = MPI_Wtime();

    worker.join();
    server.join();

    end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Tasks completed. Time left: " << end_time - start_time << "\n";
        long sumDifficulties = 0;
        for (int i = 0; i < N_TASKS; ++i) {
            sumDifficulties += task_list[i].difficulty;
        }
        std::cout << "All tasks difficulty: " << sumDifficulties / 1000000.0 << "\n";
        std::cout << "All difficulty / N: " << (sumDifficulties / 1000000.0) / size << "\n";
        std::cout << "Difference: " << std::fabs((sumDifficulties / 1000000.0) / size - (end_time - start_time)) << "\n";
    }

    MPI_Finalize();
    return 0;
}
