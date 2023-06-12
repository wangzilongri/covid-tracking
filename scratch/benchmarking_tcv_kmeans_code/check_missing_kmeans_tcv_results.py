import os

# Set the path to the directory containing the files
path = "kmeans_tcv_results"

# Create a set of expected file names
expected_files = set()

K_list = list(range(100,3100,100)) + [3136]
for K in K_list:
    for k in range(K):
        expected_files.add(f"cluster_tcv_dict_key=({K},{k}).pickle")

# Get a set of actual file names
actual_files = set(os.listdir(path))

# Find the missing files
missing_files = expected_files - actual_files

# Print the missing files, if any
present_K_set = set()
if missing_files:
    print("The following files are missing:")
    for fname in sorted(list(missing_files)):
        print(fname)
    for fname in actual_files:
        present_K = int(fname.split("(")[1].split(",")[0])
        present_K_set.add(present_K)

    print("Following K are missing")
    all_K_set = set(list(range(100,3100,100)))
    missing_K_set = all_K_set - present_K_set
    print(sorted(list(missing_K_set)))
else:
    print("All files are present.")

