#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <cuda_file.cu>"
    exit 1
fi

CUDA_FILE="$1"
EXECUTABLE="${CUDA_FILE%.*}.out"
NUM_RUNS=5

if [ ! -f "$CUDA_FILE" ]; then
    echo "Error: File '$CUDA_FILE' does not exist."
    exit 1
fi

if [[ "$CUDA_FILE" != *.cu ]]; then
    echo "Error: Input file must have .cu extension."
    exit 1
fi

echo "Compiling $CUDA_FILE with nvcc..."
nvcc -o "$EXECUTABLE" "$CUDA_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed."
    exit 1
fi

echo "Compiled $CUDA_FILE successfully."

echo "-------------------------------------"

total_time=0
total_performance=0
count=0

for ((i=1; i<=$NUM_RUNS; i++)); do
    echo -ne "Progress: $i/$NUM_RUNS\r"
    
    output=$(./"$EXECUTABLE" 2>&1)
    
    exec_time=$(echo "$output" | grep -o "Execution time: [0-9.]\+ milliseconds" | sed 's/Execution time: \([0-9.]\+\) milliseconds/\1/')
    
    performance=$(echo "$output" | grep -o "Performance: [0-9.]\+ GFlop/s" | sed 's/Performance: \([0-9.]\+\) GFlop\/s/\1/')
    
    if [[ ! -z "$exec_time" && ! -z "$performance" ]]; then
        total_time=$(echo "$total_time + $exec_time" | bc -l)
        total_performance=$(echo "$total_performance + $performance" | bc -l)
        count=$((count + 1))
    fi
done

echo -e "\n-------------------------------------"

if [ $count -gt 0 ]; then
    avg_time=$(echo "scale=4; $total_time / $count" | bc -l)
    avg_performance=$(echo "scale=4; $total_performance / $count" | bc -l)
    
    echo "Results over $count successful runs:"
    echo "Average Execution time: $avg_time milliseconds"
    echo "Average Performance: $avg_performance GFlop/s"
else
    echo "No valid metrics could be extracted from the command output."
fi

echo "Cleaning up..."
rm -f "$EXECUTABLE"
echo "Done."