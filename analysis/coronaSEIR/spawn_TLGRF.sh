#!/bin/bash

# Ensure that two arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start> <end>"
    exit 1
fi

# Get the start and end values from arguments
start=$1
end=$2

# Ensure that the start is within the valid range
if [ "$start" -lt 0 ] || [ "$start" -gt 63 ]; then
    echo "Start must be between 0 and 63."
    exit 1
fi

# Ensure that the end is within the valid range and not smaller than start
if [ "$end" -lt 0 ] || [ "$end" -gt 63 ] || [ "$end" -lt "$start" ]; then
    echo "End must be between 0 and 63, and not smaller than start."
    exit 1
fi

# Spawn parallel processes
for i in $(seq $start $end); do
  python SEIR_Changepoints_TLGRF.py $i $((i + 1)) &
done

# Wait for all background processes to complete
wait

echo "All processes completed."

