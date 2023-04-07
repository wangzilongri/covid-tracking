import os

# Set the path to the folder
path = "./individual_county_backtest_by_cutoff"

# Get a list of the existing folder names
existing_folders = [int(folder_name) for folder_name in os.listdir(path) if os.path.isdir(os.path.join(path, folder_name))]

# Define the range of folder names
folder_range = range(54, 1076)

# Find the missing folder names
missing_folders = set(folder_range) - set(existing_folders)

# Print the missing folder names
if missing_folders:
    print("The following folders are missing:")
    print(sorted(list(missing_folders)))
else:
    print("No folders are missing.")

