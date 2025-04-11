#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <cstddef>
#include <atomic>
#include <algorithm>
#include <utility>

#define N_TASKS 200
#define REQUEST_TAG 0
#define TASK_TAG 1
#define TOTAL_DIFFICULTY_CONST 2000000000

struct Task {
    int taskNumber;
    int difficulty;
    bool completed;
    double result;
};

std::mutex mtx;
std::atomic<long long> performedWeight(0);

double load_operation(const int difficulty) {
    double accum = 0.0;
    for (int i = 1; i <= difficulty; i++) {
        accum += std::sqrt(i) * std::sin(i) * std::cos(i);
    }
    return accum;
}

Task request_task(int remote_rank, const MPI_Datatype &mpi_task_type) {
    int request = 1;
    MPI_Send(&request, 1, MPI_INT, remote_rank, REQUEST_TAG, MPI_COMM_WORLD);
    Task receivedTask;
    MPI_Recv(&receivedTask, 1, mpi_task_type, remote_rank, TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return receivedTask;
}

void worker_thread(int rank, int size, std::vector<Task> &tasks, const MPI_Datatype &mpi_task_type) {
    int tasksCount = tasks.size();
    while (true) {
        Task* task_to_do = nullptr;
        for (int i = 0; i < tasksCount; ++i) {
            mtx.lock();
            if (!tasks[i].completed) {
                task_to_do = &tasks[i];
                tasks[i].completed = true;
                mtx.unlock();
                break;
            }
            mtx.unlock();
        }
        if (task_to_do != nullptr) {
            task_to_do->result = load_operation(task_to_do->difficulty);
            performedWeight.fetch_add(task_to_do->difficulty, std::memory_order_relaxed);
        } else {
            bool got_task = false;
            Task remoteTask;
            for (int i = 0; i < size; ++i) {
                if (i == rank)
                    continue;
                remoteTask = request_task(i, mpi_task_type);
                if (remoteTask.taskNumber != -1) {
                    got_task = true;
                    double res = load_operation(remoteTask.difficulty);
                    performedWeight.fetch_add(remoteTask.difficulty, std::memory_order_relaxed);
                    break;
                }
            }
            if (!got_task) {
                MPI_Barrier(MPI_COMM_WORLD);
                int dummy = 0;
                MPI_Send(&dummy, 0, MPI_INT, rank, REQUEST_TAG, MPI_COMM_WORLD);
                break;
            }
        }
    }
}

void server_thread(int rank, int size, std::vector<Task> &tasks, const MPI_Datatype &mpi_task_type) {
    int tasksCount = tasks.size();
    MPI_Status status;
    while (true) {
        int request;
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_SOURCE == rank)
            break;
        Task task_to_send;
        bool found = false;
        for (int i = 0; i < tasksCount; ++i) {
            mtx.lock();
            if (!tasks[i].completed) {
                found = true;
                tasks[i].completed = true;
                task_to_send = tasks[i];
                mtx.unlock();
                break;
            }
            mtx.unlock();
        }
        if (!found)
            task_to_send.taskNumber = -1;
        MPI_Send(&task_to_send, 1, mpi_task_type, status.MPI_SOURCE, TASK_TAG, MPI_COMM_WORLD);
    }
}


void distributionType0(int total_tasks, int size, std::vector<int>& sendcounts) {
    int base = total_tasks / size;
    int remainder = total_tasks % size;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base + (i < remainder ? 1 : 0);
    }
}

void distributionType1(int total_tasks, int size, std::vector<int>& sendcounts) {
    int sum_indices = size * (size + 1) / 2;
    int c = total_tasks / sum_indices;
    int sum_assigned = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (i + 1) * c;
        sum_assigned += sendcounts[i];
    }
    int diff = total_tasks - sum_assigned;
    int i = 0;
    while(diff > 0) {
        sendcounts[i % size]++;
        diff--;
        i++;
    }
}

void distributionType2(int total_tasks, int size, std::vector<int>& sendcounts) {
    if (size > 1) {
        sendcounts[0] = total_tasks / 2;
        int remainder = total_tasks - sendcounts[0];
        int per_other = remainder / (size - 1);
        int extra = remainder % (size - 1);
        for (int i = 1; i < size; i++) {
            sendcounts[i] = per_other + ((i - 1) < extra ? 1 : 0);
        }
    } else {
        sendcounts[0] = total_tasks;
    }
}

void create_distribution(int distributionType, int total_tasks, int size,
                         std::vector<int>& sendcounts, std::vector<int>& displs) {
    sendcounts.resize(size, 0);
    displs.resize(size, 0);
    switch(distributionType) {
        case 0:
            distributionType0(total_tasks, size, sendcounts);
            break;
        case 1:
            distributionType1(total_tasks, size, sendcounts);
            break;
        case 2:
            distributionType2(total_tasks, size, sendcounts);
            break;
        default:
            distributionType0(total_tasks, size, sendcounts);
            break;
    }
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
}

void difficultyDistributionType0(int total_tasks, std::vector<int>& difficulties) {
    int diff_const = TOTAL_DIFFICULTY_CONST / total_tasks;
    for (int i = 0; i < total_tasks; ++i) {
        difficulties[i] = diff_const;
    }
}


void difficultyDistributionType1(int total_tasks, std::vector<int>& difficulties) {
    double sum_seq = total_tasks * (total_tasks + 1) / 2.0;
    double c = TOTAL_DIFFICULTY_CONST / sum_seq;
    
    std::vector<int> tmp(total_tasks, 0);
    std::vector<double> fractions(total_tasks, 0.0);
    long long computedSum = 0;
    
    for (int i = 0; i < total_tasks; ++i) {
        double exact = (i + 1) * c;
        tmp[i] = static_cast<int>(std::floor(exact));
        computedSum += tmp[i];
        fractions[i] = exact - tmp[i];
    }
    
    int remainder = TOTAL_DIFFICULTY_CONST - computedSum;
    std::vector<std::pair<int, double>> idxFrac;
    
    for (int i = 0; i < total_tasks; ++i) {
        idxFrac.push_back({i, fractions[i]});
    }
    
    std::sort(idxFrac.begin(), idxFrac.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second > b.second;
              });
    
    for (int i = 0; i < remainder; i++) {
        tmp[idxFrac[i].first] += 1;
    }
    
    for (int i = 0; i < total_tasks; i++) {
        difficulties[i] = tmp[i];
    }
}


void difficultyDistributionType2(int total_tasks, std::vector<int>& difficulties) {
    int half = total_tasks / 2;
    
    double low_exact = static_cast<double>(TOTAL_DIFFICULTY_CONST) / (total_tasks + half);
    double high_exact = 2.0 * low_exact;
    
    std::vector<int> tmp(total_tasks, 0);
    std::vector<double> fractions(total_tasks, 0.0);
    long long computedSum = 0;
    
    for (int i = 0; i < total_tasks; ++i) {
        double exact = (i < half) ? high_exact : low_exact;
        tmp[i] = static_cast<int>(std::floor(exact));
        computedSum += tmp[i];
        fractions[i] = exact - tmp[i];
    }
    
    int remainder = TOTAL_DIFFICULTY_CONST - computedSum;
    std::vector<std::pair<int, double>> idxFrac;
    
    for (int i = 0; i < total_tasks; ++i) {
        idxFrac.push_back({i, fractions[i]});
    }
    
    std::sort(idxFrac.begin(), idxFrac.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second > b.second;
              });
    
    for (int i = 0; i < remainder; i++) {
        tmp[idxFrac[i].first] += 1;
    }
    
    for (int i = 0; i < total_tasks; i++) {
        difficulties[i] = tmp[i];
    }
}

void create_difficulty_distribution(int distributionType, int total_tasks, std::vector<int>& difficulties) {
    difficulties.resize(total_tasks);
    switch(distributionType) {
        case 0:
            difficultyDistributionType0(total_tasks, difficulties);
            break;
        case 1:
            difficultyDistributionType1(total_tasks, difficulties);
            break;
        case 2:
            difficultyDistributionType2(total_tasks, difficulties);
            break;
        default:
            difficultyDistributionType0(total_tasks, difficulties);
            break;
    }
}

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided != MPI_THREAD_MULTIPLE) {
        std::cerr << "MPI не обеспечивает необходимую поддержку потоков" << std::endl;
        MPI_Finalize();
        return -1;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype mpi_task_type;
    const int nitems = 4;
    int blocklengths[4] = {1, 1, 1, 1};
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_C_BOOL, MPI_DOUBLE};
    MPI_Aint offsets[4];
    offsets[0] = offsetof(Task, taskNumber);
    offsets[1] = offsetof(Task, difficulty);
    offsets[2] = offsetof(Task, completed);
    offsets[3] = offsetof(Task, result);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_task_type);
    MPI_Type_commit(&mpi_task_type);

    int tasksDistributionType = 2;
    int difficultyDistributionType = 0;

    std::vector<Task> allTasks;
    std::vector<int> difficulties;
    std::vector<int> sendcounts, displs;
    if (rank == 0) {
        create_difficulty_distribution(difficultyDistributionType, N_TASKS, difficulties);
        allTasks.resize(N_TASKS);
        for (int i = 0; i < N_TASKS; ++i) {
            allTasks[i].taskNumber = i;
            allTasks[i].difficulty = difficulties[i];
            allTasks[i].completed = false;
            allTasks[i].result = 0.0;
        }
        create_distribution(tasksDistributionType, N_TASKS, size, sendcounts, displs);
    }

    int local_task_count = 0;
    MPI_Scatter(rank == 0 ? sendcounts.data() : nullptr,
                1, MPI_INT,
                &local_task_count, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    std::vector<Task> localTasks(local_task_count);

    if (rank == 0) {
        MPI_Scatterv(allTasks.data(), sendcounts.data(), displs.data(), mpi_task_type,
                     localTasks.data(), local_task_count, mpi_task_type,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, mpi_task_type,
                     localTasks.data(), local_task_count, mpi_task_type,
                     0, MPI_COMM_WORLD);
    }

    long long initialWeight = 0;
    for (const auto &t : localTasks) {
        initialWeight += t.difficulty;
    }
    std::cout << "Process " << rank << " initial weight: " << initialWeight << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    std::thread worker(worker_thread, rank, size, std::ref(localTasks), mpi_task_type);
    std::thread server(server_thread, rank, size, std::ref(localTasks), mpi_task_type);

    worker.join();
    server.join();

    double end_time = MPI_Wtime();

    long long executedWeight = performedWeight.load(std::memory_order_relaxed);

    long long globalInitial = 0;
    long long globalExecuted = 0;
    MPI_Reduce(&initialWeight, &globalInitial, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&executedWeight, &globalExecuted, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Задачи выполнены. Общее время: " << (end_time - start_time) << " с" << std::endl;
        std::cout << "Глобальная сумма выполненных задач: " << globalExecuted << std::endl;
        std::cout << "Суммарный начальный вес задач: " << globalInitial << std::endl;
    }

    std::vector<long long> initWeights;
    std::vector<long long> execWeights;
    if (rank == 0) {
        initWeights.resize(size);
        execWeights.resize(size);
    }
    MPI_Gather(&initialWeight, 1, MPI_LONG_LONG,
               rank == 0 ? initWeights.data() : nullptr, 1, MPI_LONG_LONG,
               0, MPI_COMM_WORLD);
    MPI_Gather(&executedWeight, 1, MPI_LONG_LONG,
               rank == 0 ? execWeights.data() : nullptr, 1, MPI_LONG_LONG,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Данные по процессам:" << std::endl;
        for (int i = 0; i < size; ++i) {
            std::cout << "Процесс " << i << " Начальный вес: " << initWeights[i]
                      << ", финальный вес: " << execWeights[i] << std::endl;
        }
    }

    MPI_Type_free(&mpi_task_type);
    MPI_Finalize();
    return 0;
}
