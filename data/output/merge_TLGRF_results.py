import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

# Define a function to read a CSV file and return a filtered DataFrame
def read_filtered_csv(filename):
    df = pd.read_csv(filename)
    return df[pd.notnull(df['predicted.grf.future.last'])]

# Define a function to concatenate all filtered CSV files in a directory
def concat_filtered_csvs_in_directory(directory):
    filenames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    with Pool() as p:
        dfs = []
        for df in tqdm(p.imap_unordered(read_filtered_csv, filenames), total=len(filenames)):
            dfs.append(df)
    return pd.concat(dfs)

# Define the directory containing the CSV files
directory = 'confusion_state_forests_windowsize=2'

# Call the concat_filtered_csvs_in_directory function to get a single DataFrame with all rows
df = concat_filtered_csvs_in_directory(directory)

# Print the resulting DataFrame
df.to_csv("merged_TLGRF_results.csv", index=False)

