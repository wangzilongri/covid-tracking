#!/bin/bash
# Parse the command line arguments
start=$1
end=$2
step=$3

# Create the numbers array using the command line arguments
numbers=($(seq $start $step $end))

# Get the name of the server
server_name=$(hostname)

# Create the log file name
log_file=~/logs/call_parallel_tcv-${server_name}_${start}_${end}.txt
touch "$log_file"

# Loop through the numbers array and tee the output to the log file
for K in "${numbers[@]}"; do
  for k in $(seq 0 $(($K - 1))); do
    python parallel_tcv_kmeans.py $K $k 2>&1 | tee -a $log_file
  done
done




#numbers=(200)
#numbers=(3136)

# loop through list
#for K in "${numbers[@]}"; do
#for K in {1..20}; do
#  for k in $(seq 0 $(($K - 1))); do
  #for k in $(seq 1500 2000); do
#    python parallel_tcv_kmeans.py $K $k
#  done
#done

# wait for all background processes to finish
#wait

