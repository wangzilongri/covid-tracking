#!/bin/bash

# Arguments: $1 = start of the range, $2 = end of the range, $3 = step size
start=$1
end=$2
step=$3

# Loop through the range with the specified step
for (( i=$start; i<$end; i+=$step )); do
    next=$((i + step))
    if [ $next -gt $end ]; then
        next=$end
    fi
    
    # Run the python command in the background
    python STEP5_Stage2_Estimator.py $i $next &

done

# Wait for all background jobs to complete
wait
