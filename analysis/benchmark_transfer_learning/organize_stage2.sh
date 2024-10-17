#!/bin/bash

# Define the source results subfolder
results_subfolder="./stage2_betas/lambda_exp=-5"

# Extract the lambda_exp value from the results_subfolder
lambda_exp=$(basename "$results_subfolder" | sed 's/lambda_exp=//')

# Define the destination base directory for the cutoff subfolders
destination_base="./stage2_prediction"

# Loop through all .npy files in the results subfolder
for file in "$results_subfolder"/*.npy; do
    # Extract the cutoff value using parameter expansion and string manipulation
    if [[ "$file" =~ cutoff=([0-9]+) ]]; then
        cutoff="${BASH_REMATCH[1]}"  # Extract the cutoff value
        
        # Create the lambda_exp subfolder in the destination directory
        cutoff_dir="$destination_base/lambda_exp=$lambda_exp/cutoff/$cutoff"
        mkdir -p "$cutoff_dir"
        
        # Move the file into the corresponding cutoff subfolder in the new directory
        mv "$file" "$cutoff_dir/"
    fi
done

echo "Files have been organized into cutoff subfolders in the stage2_prediction directory."

