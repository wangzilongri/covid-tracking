#!/bin/bash

# Ensure the Python script is executable
#chmod +x pyscript.py

# Spawn parallel processes
for i in {0..63}; do
  start=$i
  end=$((i + 1))
  python SEIR_Changepoints_GRF.py $start $end &
done

# Wait for all background processes to complete
wait

echo "All processes completed."

