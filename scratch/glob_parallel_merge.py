import os
import glob
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Define the directory containing the subfolders with csv files
directory = '../data/output/individual_county_backtest_by_cutoff'

# Get a list of subfolders
subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]

# Define a function to read a csv file and return a pandas dataframe
def read_csv_file(filename):
    try:
        return pd.read_csv(filename)
    except:
        print(f"Error reading file: {filename}")
        return None

# Define a function to merge all CSV files in a subfolder into an intermediate CSV file
def merge_csv_files(subfolder):
    try:
        # Get the subfolder number from the folder name
        subfolder_num = os.path.basename(subfolder)

        # Get a list of CSV files in the subfolder
        csv_files = glob.glob(f"{subfolder}/*.csv")

        # Read all CSV files in the subfolder in parallel
        dfs = Parallel(n_jobs=-1)(delayed(read_csv_file)(filename) for filename in csv_files)

        # Concatenate all the dataframes
        merged_df = pd.concat([df for df in dfs if df is not None], ignore_index=True)

        # Save the merged dataframe to an intermediate CSV file
        merged_df.to_csv(f"../data/output/intermediate_by_cutoff/individual_by_cutoff={subfolder_num}.csv", index=False)
    except:
        print(f"Error merging CSV files in subfolder: {subfolder}")

# Use joblib's Parallel to merge all subfolders' CSV files in parallel
with tqdm(total=len(subfolders), desc="Merging CSV files") as pbar:
    Parallel(n_jobs=-1)(delayed(merge_csv_files)(subfolder) for subfolder in subfolders)
    pbar.update(1)

