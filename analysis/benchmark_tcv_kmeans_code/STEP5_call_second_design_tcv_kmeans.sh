#!/bin/bash

#numbers=(200)
#numbers=(3136)
start=$1
end=$2
step=$3

numbers=($(seq $start $step $end))

server_name=$(hostname)
log_file=~/logs/call_second_design_parallel_tcv-${server_name}_${start}_${end}.txt
touch "$log_file"

# loop through list
for K in "${numbers[@]}"; do
#for K in {20..20}; do
  #for k in $(seq 1500 2000); do
  python second_design_tcv_kmeans.py $K >> $log_file 2>&1 &
done
