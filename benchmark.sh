#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command_to_run>"
    exit 1
fi

CMD="$@"
NUM_RUNS=100

total_time=0
total_performance=0
count=0

echo "Running $CMD $NUM_RUNS times..."
echo "-------------------------------------"

for ((i=1; i<=$NUM_RUNS; i++)); do
    echo -ne "Progress: $i/$NUM_RUNS\r"
    
    output=$($CMD 2>&1)
    
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
