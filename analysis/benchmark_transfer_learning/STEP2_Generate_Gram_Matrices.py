#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ast
import concurrent.futures
import glob
import itertools
import os
import pickle
import warnings
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator


import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.cluster import KMeans

from statsmodels.regression.rolling import RollingOLS

from tqdm.notebook import tqdm
from collections import Counter
from functools import reduce
from pprint import pprint

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.6f}'.format)


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ### Read in parquets

# In[2]:


scaled_numeric_even_combined_df = pl.read_parquet("scaled_numeric_even_combined_df.parquet").sort(["cutoff","fips"])
scaled_numeric_odd_combined_df = pl.read_parquet("scaled_numeric_odd_combined_df.parquet").sort(["cutoff","fips"])


# In[3]:


even_indices = scaled_numeric_even_combined_df.select(["cutoff"]).unique().sort(["cutoff"])
odd_indices = scaled_numeric_odd_combined_df.select(["cutoff"]).unique().sort(["cutoff"])


# In[4]:


odd_indices


# In[5]:


# Columns to exclude
index_cols = ["fips", "cutoff"]
exclude_cols = ["State_FIPS_Code", "log_rolled_cases.x", "log_rolled_cases.y", "t0.lm", "r.lm"]
outcome_col = "shifted_log_rolled_cases"
feature_cols = list(set(scaled_numeric_even_combined_df.columns) - set(index_cols) - set(exclude_cols) - set(outcome_col))

os.makedirs("gram_matrices", exist_ok=True)

# In[6]:


# Function to check if backup file already exists
def check_existing_backup(cutoff):
    filename_XtX = f"gram_matrices/gram_matrix_cutoff={cutoff}.npy"
    filename_Xty = f"gram_matrices/Xty_cutoff={cutoff}.npy"
    filename_y = f"gram_matrices/y_cutoff={cutoff}.npy"
    
    return os.path.exists(filename_XtX) and os.path.exists(filename_y)

# Function to determine whether a cutoff is odd or even
def is_even(cutoff):
    return cutoff % 2 == 0

def generate_XtX_y(index, even_df, odd_df, even_indices, odd_indices):
    cutoff = index['cutoff']
    
    # Select the corresponding dataset based on whether the fips is even or odd
    if is_even(cutoff):
        dataset = even_df
        indices_df = even_indices
    else:
        dataset = odd_df
        indices_df = odd_indices

    # Filter data up to the current cutoff and exclude the current index's data point
    filtered_data = dataset.filter(pl.col("cutoff") <= cutoff)

    # Select X (features) and y (outcome)
    X = filtered_data.drop(index_cols + exclude_cols + [outcome_col]).with_columns(pl.lit(1).alias("intercept"))
    y = filtered_data.select(outcome_col)

    # Convert to NumPy for scikit-learn compatibility
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()

    XtX = np.dot(X_np.T, X_np)

    save_results_to_csv(cutoff, XtX, y_np)
    return True

def generate_XtX_y_with_error_handling(index, even_df, odd_df, even_indices, odd_indices):
    cutoff = index["cutoff"]
    if check_existing_backup(cutoff):
        tqdm.write("cutoff={} exists".format(cutoff))
        return None
    try:
        # Call the original train_ols function
        return generate_XtX_y(index, even_df, odd_df, even_indices, odd_indices)
    except Exception as e:
        # If any error occurs, print the error and return None
        tqdm.write(f"Error occurred while processing cutoff={index['cutoff']}: {e}")
        return None

def save_results_to_csv(cutoff, XtX, y):    
    filename_XtX = f"gram_matrices/gram_matrix_cutoff={cutoff}.npy"
    filename_y = f"gram_matrices/y_cutoff={cutoff}.npy"
    tqdm.write("Saving {} and {}".format(filename_XtX, filename_y))
    # Save the DataFrame as a CSV file
    np.save(filename_XtX, XtX)
    np.save(filename_y, y)
    return

# Function to run the process in parallel
def run_parallel(even_df, odd_df, even_indices, odd_indices, start_cutoff=50, end_cutoff=1200):
    # Combine both even and odd indices into one list
    indices = even_indices.to_dicts() + odd_indices.to_dicts()

    # Filter indices based on the provided cutoff range
    indices = ([index for index in indices if start_cutoff <= index['cutoff'] <= end_cutoff])
    
    results = []

    # Add tqdm for progress bar
    with ThreadPoolExecutor() as executor, tqdm(total=len(indices), desc="Generating XtX and y from t={}:{}".format(start_cutoff, end_cutoff)) as pbar:
        futures = []

        for index in indices:
            cutoff = int(index['cutoff'])
            
            futures.append(executor.submit(generate_XtX_y_with_error_handling, index, even_df, odd_df, even_indices, odd_indices))
        
        # Collect results and update the progress bar
        for future in futures:
            pbar.update(1)
    
    return results


# In[ ]:


# Main function to handle sysargs for start and end cutoff
if __name__ == "__main__":
    # Read command-line arguments for start and end cutoff
    start_cutoff = 50
    end_cutoff = 1200
    if len(sys.argv) != 3:
        print("Usage: python STEP3_Generate_Gram_Matrices.py <start_cutoff> <end_cutoff>")
        print("Defaulting to start_cutoff := 50, end_cutoff := 1200")
    else:
        start_cutoff = int(sys.argv[1])
        end_cutoff = int(sys.argv[2])

    # Example usage:
    # even_indices and odd_indices should be provided as polars DataFrames
    # numeric_even_combined_df and numeric_odd_combined_df should be the corresponding datasets

    # Call the parallel OLS training for even and odd datasets within the cutoff range
    stage1_results = run_parallel(scaled_numeric_even_combined_df, scaled_numeric_odd_combined_df, even_indices, odd_indices, start_cutoff, end_cutoff)
    #indices = even_indices.to_dicts() + odd_indices.to_dicts()

    # Filter indices based on the provided cutoff range
    #indices = ([index for index in indices if start_cutoff <= index['cutoff'] <= end_cutoff])
    #for index in tqdm(total=len(indices), desc="Generating XtX and y from t={}:{}".format(start_cutoff, end_cutoff)):
    #    generate_XtX_y_with_error_handling(index, scaled_numeric_even_combined_df, scaled_numeric_odd_combined_df, even_indices, odd_indices)


# In[ ]:




