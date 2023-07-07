#!/bin/bash


start=$1
end=$2
step=$3

server_name=$(hostname)
log_file=~/logs/call_cv_subroutine/call_cv_subroutine-${server_name}_${start}_${end}.txt
touch "$log_file"

#numbers=($(seq $start $step $end))
# loop through list
#for K in "${numbers[@]}"; do
#for K in {20..20}; do
  #for k in $(seq 1500 2000); do
  #python cv_subroutine_per_cluster.py $K 2>&1 | tee -a $log_file
#done
python cv_subroutine_per_cluster.py $1 $2 $3 2>&1 | tee -a $log_file

