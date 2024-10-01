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

from sklearn.linear_model import LinearRegression, Ridge
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


# ### Read in Odd and Even Stitched Blocks

# In[2]:
print("Reading in stitched blocks and generating indices")

scaled_numeric_even_combined_df = pl.read_parquet("scaled_numeric_even_combined_df.parquet").sort(["cutoff","fips"])
scaled_numeric_odd_combined_df = pl.read_parquet("scaled_numeric_odd_combined_df.parquet").sort(["cutoff","fips"])

even_indices = scaled_numeric_even_combined_df.select(["cutoff"]).unique().sort(["cutoff"])
odd_indices = scaled_numeric_odd_combined_df.select(["cutoff"]).unique().sort(["cutoff"])


# In[5]:


### For each (t,c) 
### Train estimator shifted_log_rolled_cases_{-(t,c)} = beta * x_{-t,c} + intercept 


# In[6]:


# Columns to exclude
index_cols = ["fips", "cutoff"]
exclude_cols = ["State_FIPS_Code", "log_rolled_cases.x", "log_rolled_cases.y", "t0.lm", "r.lm"]
outcome_col = "shifted_log_rolled_cases"
feature_cols = list(set(scaled_numeric_even_combined_df.columns) - set(index_cols) - set(exclude_cols) - set(outcome_col))
#print(feature_cols)
# Create the "stage1_results" directory if it doesn't exist
os.makedirs("approximate_stage1_results", exist_ok=True)


# In[ ]:





# In[7]:


# Function to check if backup file already exists
def backup_exists(cutoff):
    filename = f"approximate_stage1_results/stage1_cutoff={cutoff}.csv"
    return os.path.exists(filename)

# Function to determine whether a fips is odd or even
def is_even(fips_value):
    return fips_value % 2 == 0

# Function to prepare data, normalize, train OLS estimator, and return results
def train_ols(index, even_df, odd_df, even_indices, odd_indices):
    cutoff = index['cutoff']
    
    # Select the corresponding dataset based on whether the fips is even or odd
    if is_even(cutoff):
        dataset = even_df
        indices_df = even_indices
    else:
        dataset = odd_df
        indices_df = odd_indices

    # Filter data up to the current cutoff
    filtered_data = dataset.filter(pl.col("cutoff") <= cutoff)#.filter(pl.col("fips") != fips)

    # Select X (features) and y (outcome)
    X = filtered_data.drop(index_cols + exclude_cols + [outcome_col]).with_columns(pl.lit(1).alias("intercept"))
    y = filtered_data.select(outcome_col)

    # Convert to NumPy for scikit-learn compatibility
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()

    # Train OLS model
    model = Ridge(alpha=0, fit_intercept=False)  # Already added intercept manually
    model.fit(X_np, y_np)

    beta = model.coef_

    # Return the result with fips, cutoff, and the learned beta coefficients
    result = {
        "cutoff": cutoff,
        "beta": beta
    }
    # Save each result to a CSV file in the "stage1_results" directory
    save_results_to_csv(result)

    return result

# Helper function to train OLS with error handling
def train_ols_with_error_handling(index, even_df, odd_df, even_indices, odd_indices):
    cutoff = index["cutoff"]
    if backup_exists(cutoff):
        tqdm.write("cutoff={} exists".format(cutoff))
        return None
    try:
        # Call the original train_ols function
        return train_ols(index, even_df, odd_df, even_indices, odd_indices)
    except Exception as e:
        # If any error occurs, print the error and return None
        tqdm.write(f"Error occurred while processing fips={index['fips']}, cutoff={index['cutoff']}: {e}")
        return None


# Function to save results to CSV
def save_results_to_csv(result):
    cutoff = result["cutoff"]
    
    # Create a DataFrame from the result
    beta_df = pl.DataFrame({
        "beta": result["beta"]
    })
    
    # Define the CSV file name
    filename = f"approximate_stage1_results/stage1_cutoff={cutoff}.csv"
    tqdm.write("Saving {}".format(filename))
    # Save the DataFrame as a CSV file
    beta_df.write_csv(filename)
    
# Function to run the process in parallel
def run_parallel_ols(even_df, odd_df, even_indices, odd_indices, start_cutoff=50, end_cutoff=1200):
    # Combine both even and odd indices into one list
    indices = even_indices.to_dicts() + odd_indices.to_dicts()

    # Filter indices based on the provided cutoff range
    indices = [index for index in indices if start_cutoff <= index['cutoff'] <= end_cutoff]
    
    results = []

    # Add tqdm for progress bar
    # ThreadPoolExecutor(max_workers=30)
    with ThreadPoolExecutor(max_workers=10) as executor, tqdm(total=len(indices), desc="Training OLS models from t={}:{}".format(start_cutoff, end_cutoff)) as pbar:
        futures = []

        for index in indices:
            cutoff = int(index['cutoff'])
            
            futures.append(executor.submit(train_ols_with_error_handling, index, even_df, odd_df, even_indices, odd_indices))
        
        # Collect results and update the progress bar
        for future in futures:
            if future.result() is not None:
                results.append(future.result())
            pbar.update(1)
    
    return results

# Function to convert results into a Polars DataFrame and save as Parquet
def save_as_parquet(results, filename):
    # Convert to a DataFrame with fips, cutoff, and beta columns
    df = pl.DataFrame({
        "cutoff": [res["cutoff"] for res in results],
        "beta": [res["beta"] for res in results]
    })

    df.write_parquet(filename)


# In[ ]:


# Main function to handle sysargs for start and end cutoff
if __name__ == "__main__":
    # Read command-line arguments for start and end cutoff
    start_cutoff = 50
    end_cutoff = 1200

    if len(sys.argv) != 3:
        print("Usage: python Alternate_STEP2_Generate_Stage1_Beta.py <start_cutoff> <end_cutoff>")
        print("Defaulting to start_cutoff := 50, end_cutoff := 1200")
    else:
        start_cutoff = int(sys.argv[1])
        end_cutoff = int(sys.argv[2])

    # Example usage:
    # even_indices and odd_indices should be provided as polars DataFrames
    # numeric_even_combined_df and numeric_odd_combined_df should be the corresponding datasets

    # Call the parallel OLS training for even and odd datasets within the cutoff range

    stage1_results = run_parallel_ols(scaled_numeric_even_combined_df, scaled_numeric_odd_combined_df, even_indices, odd_indices, start_cutoff, end_cutoff)
    save_as_parquet(stage1_results, "alternate_stage1_results.parquet")
    
    #indices = even_indices.to_dicts() + odd_indices.to_dicts()

    # Filter indices based on the provided cutoff range
    #indices = [index for index in indices if start_cutoff <= index['cutoff'] <= end_cutoff]

    #for index in indices:
    #    train_ols_with_error_handling(index, scaled_numeric_even_combined_df, scaled_numeric_odd_combined_df, even_indices, odd_indices)

# In[ ]:





# In[ ]:




