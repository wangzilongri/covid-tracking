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

from sklearn.linear_model import LinearRegression, ridge_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import joblib


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


# In[3]:


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
feature_cols = list(set(scaled_numeric_even_combined_df.columns) - set(index_cols) - set(exclude_cols) - set([outcome_col]))
#print(feature_cols)
# Create the "stage1_results" directory if it doesn't exist
os.makedirs("stage1_lasso", exist_ok=True)


# In[ ]:





# In[7]:


# Function to check if backup file already exists
def backup_exists(cutoff):
    fname = f'./stage1_lasso/best_lasso_model_cutoff={cutoff}.pkl'
    return os.path.exists(fname)

# Function to determine whether a cutoff is odd or even
def is_even(cutoff):
    return cutoff % 2 == 0

# Function to prepare data, normalize, train OLS estimator, and return results
def train_ols(index, even_df, odd_df):
    cutoff = index['cutoff']

    if is_even(cutoff):
        dataset = odd_df
    else:
        dataset = even_df
    
    # Filter data up to the current cutoff
    filtered_data = dataset.filter(pl.col("cutoff") <= cutoff)#.filter(pl.col("fips") != fips)

    # Select X (features) and y (outcome)
    X = filtered_data.drop(index_cols + exclude_cols + [outcome_col])#.with_columns(pl.lit(1).alias("intercept"))
    y = filtered_data.select(outcome_col)

    # Convert to NumPy for scikit-learn compatibility
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()

    # Train OLS model
    #model = Ridge(alpha=0, fit_intercept=False)  # Already added intercept manually
    #model.fit(X_np, y_np)
    #X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

    lasso = Lasso(fit_intercept=True)  # This is the default setting
    param_grid = {'alpha': [10**i for i in range(-5,5)]}
    
    # 5. Perform GridSearchCV to find the best L1 penalty
    grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_np, y_np)

    best_lasso = grid_search.best_estimator_
    
    y_pred = best_lasso.predict(X_np)
    
    mse_full = mean_squared_error(y_np, y_pred)
        
    print(f"Best Alpha: {grid_search.best_params_['alpha']}")
    print(f"Test MSE: {mse_full}")

    print(f"Saving y_pred of best model of cutoff={cutoff}")
    np.save(f'./stage1_lasso/y_pred_cutoff={cutoff}.npy', y_pred)

    # 8. Save the best model to a file
    fname = f'./stage1_lasso/best_lasso_model_cutoff={cutoff}.pkl'
    print(f"Saving model to {fname}")
    joblib.dump(best_lasso, fname)

    return True

# Helper function to train OLS with error handling
def train_ols_with_error_handling(index, even_df, odd_df):
    cutoff = index["cutoff"]

    if backup_exists(cutoff):
        tqdm.write("cutoff={} exists".format(cutoff))
        return None
    try:
        # Call the original train_ols function
        return train_ols(index, even_df, odd_df)
    except Exception as e:
        # If any error occurs, print the error and return None
        tqdm.write(f"Error occurred while processing cutoff={index['cutoff']}: {e}")
        return None


# Function to save results to npy
def save_results_to_npy(result):
    cutoff = result["cutoff"]
    
    # Create a DataFrame from the result
    beta = result["beta"]
    
    # Define the CSV file name
    filename = f"betas_penalized/beta_cutoff={cutoff}.npy"
    tqdm.write("Saving {}".format(filename))
    np.save(filename, beta)
    
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
            
            futures.append(executor.submit(train_ols_with_error_handling, index, even_df, odd_df))
        
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
    #save_as_parquet(stage1_results, "alternate_stage1_results.parquet")
    
    #indices = even_indices.to_dicts() + odd_indices.to_dicts()

    # Filter indices based on the provided cutoff range
    #indices = [index for index in indices if start_cutoff <= index['cutoff'] <= end_cutoff]

    #for index in indices:
    #    train_ols_with_error_handling(index, scaled_numeric_even_combined_df, scaled_numeric_odd_combined_df, even_indices, odd_indices)

# In[ ]:





# In[ ]:




