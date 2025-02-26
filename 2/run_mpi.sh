#!/bin/bash

P=5
NP=8
maxIterations=150
N=15000

min_time=""

for ((i=1; i<=P; i++)); do
    echo "Запуск #$i NP=$NP"
    output=$(mpirun -np $NP ./mpi_var2 $maxIterations $N)

    echo "$output"
    current_time=$(echo "$output" | grep "Elapsed time:" | awk '{print $3}')

    if [[ -z "$min_time" ]] || (( $(echo "$current_time < $min_time" | bc -l) )); then
        min_time=$current_time
    fi
done

echo "Минимальное время работы: $min_time"
