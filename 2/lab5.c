#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stddef.h>
#include <pthread.h>
#include <stdatomic.h>
#include <mpi.h>

#define N_TASKS 200
#define REQUEST_TAG 0
#define TASK_TAG 1
#define TOTAL_DIFFICULTY_CONST 2000000000LL

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int taskNumber;
    int difficulty;
    bool completed;
    double result;
} Task;

pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
atomic_llong performedWeight = 0;

double load_operation(const int difficulty) {
    double accum = 0.0;
    for (int i = 1; i <= difficulty; i++) {
        accum += sqrt(i) * sin(i) * cos(i);
    }
    return accum;
}

Task request_task(int remote_rank, MPI_Datatype mpi_task_type) {
    int request = 1;
    MPI_Send(&request, 1, MPI_INT, remote_rank, REQUEST_TAG, MPI_COMM_WORLD);
    Task receivedTask;
    MPI_Recv(&receivedTask, 1, mpi_task_type, remote_rank, TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return receivedTask;
}

typedef struct {
    int rank;
    int size;
    Task *tasks;
    int tasksCount;
    MPI_Datatype mpi_task_type;
} thread_args_t;


void *worker_thread(void *arg) {
    thread_args_t *args = (thread_args_t*) arg;
    int rank = args->rank;
    int size = args->size;
    Task *tasks = args->tasks;
    int tasksCount = args->tasksCount;
    MPI_Datatype mpi_task_type = args->mpi_task_type;

    while (1) {
        Task *task_to_do = NULL;
        // Поиск невыполненной локальной задачи
        for (int i = 0; i < tasksCount; i++) {
            pthread_mutex_lock(&mtx);
            if (!tasks[i].completed) {
                task_to_do = &tasks[i];
                tasks[i].completed = true;
                pthread_mutex_unlock(&mtx);
                break;
            }
            pthread_mutex_unlock(&mtx);
        }
        if (task_to_do != NULL) {
            task_to_do->result = load_operation(task_to_do->difficulty);
            atomic_fetch_add(&performedWeight, task_to_do->difficulty);
        } else {
            int got_task = 0;
            Task remoteTask;
            for (int i = 0; i < size; i++) {
                if (i == rank)
                    continue;
                remoteTask = request_task(i, mpi_task_type);
                if (remoteTask.taskNumber != -1) {
                    got_task = 1;
                    double res = load_operation(remoteTask.difficulty);
                    (void)res; // результат можно использовать по необходимости
                    atomic_fetch_add(&performedWeight, remoteTask.difficulty);
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
    return NULL;
}

void server_thread(int rank, int size, Task *tasks, int tasksCount, MPI_Datatype mpi_task_type) {
    MPI_Status status;
    while (1) {
        int request;
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_SOURCE == rank)
            break;
        Task task_to_send;
        int found = 0;
        for (int i = 0; i < tasksCount; i++) {
            pthread_mutex_lock(&mtx);
            if (!tasks[i].completed) {
                found = 1;
                tasks[i].completed = true;
                task_to_send = tasks[i];
                pthread_mutex_unlock(&mtx);
                break;
            }
            pthread_mutex_unlock(&mtx);
        }
        if (!found)
            task_to_send.taskNumber = -1;
        MPI_Send(&task_to_send, 1, mpi_task_type, status.MPI_SOURCE, TASK_TAG, MPI_COMM_WORLD);
    }
}

//равномерное распределение
void distributionType0(int total_tasks, int size, int *sendcounts) {
    int base = total_tasks / size;
    int remainder = total_tasks % size;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base + (i < remainder ? 1 : 0);
    }
}

//возрастающая горка
void distributionType1(int total_tasks, int size, int *sendcounts) {
    int sum_indices = size * (size + 1) / 2;
    double c = (double)total_tasks / (double)sum_indices;
    int sum_assigned = 0;
    for (int i = 0; i < size; i++) {
        double exact = (i + 1) * c;
        sendcounts[i] = (int)floor(exact);
        sum_assigned += sendcounts[i];
    }
    int diff = total_tasks - sum_assigned;
    int i = 0;
    while (diff > 0) {
        sendcounts[i % size]++;
        diff--;
        i++;
    }
}


void distributionType2(int total_tasks, int size, int *sendcounts) {
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

//Синус
void distributionType3(int total_tasks, int size, int *sendcounts) {
    double sum = 0.0;
    double *weights = (double*) malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        weights[i] = sin((i + 1) * M_PI / (size + 1));
        sum += weights[i];
    }

    int *temp = (int*) malloc(size * sizeof(int));
    int assigned = 0;
    for (int i = 0; i < size; i++) {
        temp[i] = (int) floor((weights[i] / sum) * total_tasks);
        assigned += temp[i];
    }
    int remainder = total_tasks - assigned;

    double *fractions = (double*) malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        double ideal = (weights[i] / sum) * total_tasks;
        fractions[i] = ideal - temp[i];
    }

    int *indices = (int*) malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        indices[i] = i;
    }
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            if (fractions[indices[j]] > fractions[indices[i]]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }
    for (int i = 0; i < remainder; i++) {
        temp[indices[i]] += 1;
    }
    for (int i = 0; i < size; i++) {
        sendcounts[i] = temp[i];
    }

    free(weights);
    free(temp);
    free(fractions);
    free(indices);
}

//Все задачи у одного процесса (0)
void distributionType4(int total_tasks, int size, int *sendcounts) {
    sendcounts[0] = total_tasks;
    for (int i = 1; i < size; i++) {
        sendcounts[i] = 0;
    }
}

void create_distribution(int distributionType, int total_tasks, int size, int *sendcounts, int *displs) {
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
        case 3:
            distributionType3(total_tasks, size, sendcounts);
            break;
        case 4:
            distributionType4(total_tasks, size, sendcounts);
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



void difficultyDistributionType0(int total_tasks, int *difficulties) {
    int diff_const = TOTAL_DIFFICULTY_CONST / total_tasks;
    for (int i = 0; i < total_tasks; i++) {
        difficulties[i] = diff_const;
    }
}

typedef struct {
    int index;
    double fraction;
} IndexFraction;

void difficultyDistributionType1(int total_tasks, int *difficulties) {
    double sum_seq = total_tasks * (total_tasks + 1) / 2.0;
    double c = (double)TOTAL_DIFFICULTY_CONST / sum_seq;
    int *tmp = (int*)malloc(total_tasks * sizeof(int));
    double *fractions = (double*)malloc(total_tasks * sizeof(double));
    long long computedSum = 0;
    for (int i = 0; i < total_tasks; i++) {
        double exact = (i + 1) * c;
        tmp[i] = (int)floor(exact);
        computedSum += tmp[i];
        fractions[i] = exact - tmp[i];
    }
    int remainder = TOTAL_DIFFICULTY_CONST - computedSum;
    IndexFraction *idxFrac = (IndexFraction*)malloc(total_tasks * sizeof(IndexFraction));
    for (int i = 0; i < total_tasks; i++) {
        idxFrac[i].index = i;
        idxFrac[i].fraction = fractions[i];
    }
    int compare(const void *a, const void *b) {
        double diff = ((IndexFraction*)b)->fraction - ((IndexFraction*)a)->fraction;
        return (diff > 0) ? 1 : (diff < 0 ? -1 : 0);
    }
    qsort(idxFrac, total_tasks, sizeof(IndexFraction), compare);
    for (int i = 0; i < remainder; i++) {
        tmp[idxFrac[i].index] += 1;
    }
    for (int i = 0; i < total_tasks; i++) {
        difficulties[i] = tmp[i];
    }
    free(tmp);
    free(fractions);
    free(idxFrac);
}

void difficultyDistributionType2(int total_tasks, int *difficulties) {
    int half = total_tasks / 2;
    double low_exact = (double)TOTAL_DIFFICULTY_CONST / (total_tasks + half);
    double high_exact = 2.0 * low_exact;
    int *tmp = (int*)malloc(total_tasks * sizeof(int));
    double *fractions = (double*)malloc(total_tasks * sizeof(double));
    long long computedSum = 0;
    for (int i = 0; i < total_tasks; i++) {
        double exact = (i < half) ? high_exact : low_exact;
        tmp[i] = (int)floor(exact);
        computedSum += tmp[i];
        fractions[i] = exact - tmp[i];
    }
    int remainder = TOTAL_DIFFICULTY_CONST - computedSum;
    IndexFraction *idxFrac = (IndexFraction*)malloc(total_tasks * sizeof(IndexFraction));
    for (int i = 0; i < total_tasks; i++) {
        idxFrac[i].index = i;
        idxFrac[i].fraction = fractions[i];
    }
    int compare(const void *a, const void *b) {
        double diff = ((IndexFraction*)b)->fraction - ((IndexFraction*)a)->fraction;
        return (diff > 0) ? 1 : (diff < 0 ? -1 : 0);
    }
    qsort(idxFrac, total_tasks, sizeof(IndexFraction), compare);
    for (int i = 0; i < remainder; i++) {
        tmp[idxFrac[i].index] += 1;
    }
    for (int i = 0; i < total_tasks; i++) {
        difficulties[i] = tmp[i];
    }
    free(tmp);
    free(fractions);
    free(idxFrac);
}

void create_difficulty_distribution(int distributionType, int total_tasks, int *difficulties) {
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
        fprintf(stderr, "MPI does not provide needed thread support\n");
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

    int tasksDistributionType = 0;
    int difficultyDistributionType = 0;

    Task *allTasks = NULL;
    int *difficulties = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        difficulties = (int*) malloc(N_TASKS * sizeof(int));
        sendcounts = (int*) malloc(size * sizeof(int));
        displs = (int*) malloc(size * sizeof(int));
        create_difficulty_distribution(difficultyDistributionType, N_TASKS, difficulties);
        allTasks = (Task*) malloc(N_TASKS * sizeof(Task));
        for (int i = 0; i < N_TASKS; i++) {
            allTasks[i].taskNumber = i;
            allTasks[i].difficulty = difficulties[i];
            allTasks[i].completed = false;
            allTasks[i].result = 0.0;
        }
        create_distribution(tasksDistributionType, N_TASKS, size, sendcounts, displs);
    }

    int local_task_count = 0;
    MPI_Scatter(rank == 0 ? sendcounts : NULL, 1, MPI_INT,
                &local_task_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Task *localTasks = (Task*) malloc(local_task_count * sizeof(Task));
    MPI_Scatterv(rank == 0 ? allTasks : NULL, sendcounts, displs, mpi_task_type,
                 localTasks, local_task_count, mpi_task_type, 0, MPI_COMM_WORLD);

    long long initialWeight = 0;
    for (int i = 0; i < local_task_count; i++) {
        initialWeight += localTasks[i].difficulty;
    }
    printf("Process %d initial weight: %lld\n", rank, initialWeight);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    pthread_t worker;
    thread_args_t args;
    args.rank = rank;
    args.size = size;
    args.tasks = localTasks;
    args.tasksCount = local_task_count;
    args.mpi_task_type = mpi_task_type;
    pthread_create(&worker, NULL, worker_thread, &args);

    server_thread(rank, size, localTasks, local_task_count, mpi_task_type);

    pthread_join(worker, NULL);
    double end_time = MPI_Wtime();

    long long executedWeight = atomic_load(&performedWeight);

    long long globalInitial = 0;
    long long globalExecuted = 0;
    MPI_Reduce(&initialWeight, &globalInitial, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&executedWeight, &globalExecuted, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Tasks completed. Total time: %f s\n", (end_time - start_time));
        printf("Global executed weight: %lld\n", globalExecuted);
        printf("Global initial weight: %lld\n", globalInitial);
    }

    long long *initWeights = NULL;
    long long *execWeights = NULL;
    if (rank == 0) {
        initWeights = (long long*) malloc(size * sizeof(long long));
        execWeights = (long long*) malloc(size * sizeof(long long));
    }
    MPI_Gather(&initialWeight, 1, MPI_LONG_LONG,
               rank == 0 ? initWeights : NULL, 1, MPI_LONG_LONG,
               0, MPI_COMM_WORLD);
    MPI_Gather(&executedWeight, 1, MPI_LONG_LONG,
               rank == 0 ? execWeights : NULL, 1, MPI_LONG_LONG,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Data per process:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: Initial weight: %lld, Executed weight: %lld\n", i, initWeights[i], execWeights[i]);
        }
    }

    free(localTasks);
    if (rank == 0) {
        free(allTasks);
        free(difficulties);
        free(sendcounts);
        free(displs);
        free(initWeights);
        free(execWeights);
    }

    MPI_Type_free(&mpi_task_type);
    MPI_Finalize();
    return 0;
}
