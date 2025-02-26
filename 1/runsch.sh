#!/bin/bash

# Проверяем, что передано ровно четыре аргумента
if [ "$#" -ne 4 ]; then
  echo "Использование: $0 <режим (1 или 2)> <N> <maxIterations> <число запусков>"
  exit 1
fi

# Параметры для программы
prog_mode=$1
prog_N=$2
prog_maxIter=$3
num_runs=$4

# Компиляция исходников с поддержкой OpenMP
g++ -fopenmp -O3 lab1_omp_var1.cpp matrix_generators_omp.cpp -o lab1_omp
if [ $? -ne 0 ]; then
  echo "Ошибка компиляции"
  exit 1
fi

min_time=1000000000

export OMP_NUM_THREADS=4

# Запуск программы num_runs раз
for ((i=1; i<=num_runs; i++)); do
  echo "Запуск $i:"
  output=$(./lab1_omp "$prog_mode" "$prog_N" "$prog_maxIter")
  echo "$output"

  # Извлекаем время выполнения из строки "Время выполнения: ..."
  run_time=$(echo "$output" | grep "Время выполнения:" | awk '{print $(NF-1)}')

  if [ -z "$run_time" ]; then
    echo "Не удалось извлечь время выполнения из запуска $i"
    continue
  fi

  cmp=$(echo "$run_time < $min_time" | bc -l)
  if [ "$cmp" -eq 1 ]; then
    min_time=$run_time
  fi
done

echo "Минимальное время выполнения: $min_time секунд."
