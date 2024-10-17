#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ast
import concurrent.futures
import glob
import itertools
import joblib
import os
import pickle
import warnings
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator

import cvxpy as cp
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

from tqdm import tqdm
from collections import Counter
from functools import reduce
from pprint import pprint

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.6f}'.format)


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ### Import dataset and betas

# In[2]:


# Dataset
combined_data = pl.read_parquet("./scaled_numeric_combined_df.parquet").sort(["cutoff","fips","shifted_time"])
# Columns to exclude
index_cols = ["fips", "cutoff"]
exclude_cols = ["State_FIPS_Code", "log_rolled_cases.x", "log_rolled_cases.y", "t0.lm", "r.lm"]
outcome_col = "shifted_log_rolled_cases"
feature_cols = list(set(combined_data.columns) - set(index_cols) - set(exclude_cols) - set([outcome_col]))
# Indices
indices = combined_data.select(index_cols).sort(["cutoff","fips"])

cutoff_list = sorted(indices.select(["cutoff"]).unique().to_pandas().values.reshape(-1))
fips_list = sorted(indices.select(["fips"]).unique().to_pandas().values.reshape(-1))
lambda_exps = range(-2, 5)


# In[3]:


#cutoff_list = range(500,510)
fips_list = sorted(indices.select(["fips"]).unique().to_pandas().values.reshape(-1))
lambda_exps = range(-5, -3)


# In[4]:


#fips_list


# ### Load betas
# - Load a later day so that `beta_{t}` and `beta_{-t,c}` don't differ that much

# In[5]:


#t = 600
#c = 2240


# ### Test `X_{t,c}`

# In[8]:


os.makedirs("./stage2_betas_organized", exist_ok=True)
for lambda_exp in lambda_exps:
    os.makedirs(f"./stage2_betas_organized/lambda_exp={int(lambda_exp)}", exist_ok=True)
    for cutoff in cutoff_list:
        os.makedirs(f"./stage2_betas_organized/lambda_exp={int(lambda_exp)}/{cutoff}", exist_ok=True)



# In[9]:


lambda_reg = 1.0  # Regularization parameter

def slice_data(combined_data, t, c):
    x_tc = combined_data.filter(pl.col("cutoff") == t).filter(pl.col("fips")==c)
    y_tc = x_tc.select(outcome_col).to_pandas().values
    x_tc = x_tc.select(feature_cols).with_columns(pl.lit(1).alias("Intercept")).to_pandas().values

    return x_tc, y_tc

def stage2_beta(stage1_beta_with_intercept, x_tc, y_tc, t, c, lambda_exp):
    lambda_reg = 10**lambda_exp

    # Define variable for the optimization (beta to be optimized)
    beta_tc = cp.Variable(stage1_beta_with_intercept.shape[0])
    
    predicted = cp.reshape(x_tc @ beta_tc, (2, 1))
    
    # Define the RMSE term: (1/n) * || X @ beta - y ||_2
    rmse_term = cp.norm(predicted - y_tc, 2)**2 / (y_tc.shape[0])
    
    # Define the regularization term: lambda * || beta - stage1_beta ||_1
    reg_term = lambda_reg * cp.norm(beta_tc - stage1_beta_with_intercept, 1)
    
    # Objective function: minimize RMSE + lambda * regularization term
    objective = cp.Minimize(rmse_term + reg_term)
    
    # Problem setup
    problem = cp.Problem(objective)
    
    # Solve the problem
    problem.solve(), beta_tc.value

    # Save stage2_beta
    fname = f"./stage2_betas_organized/lambda_exp={int(lambda_exp)}/{t}/stage2_beta_lambda_exp={int(lambda_exp)}_cutoff={t}_fips={c}.npy"
    tqdm.write("Saving {}".format(fname))
    np.save(fname, beta_tc.value)
    
    y_pred_fname = f"./stage2_betas_organized/lambda_exp={int(lambda_exp)}/{t}/stage2_y_pred_exp={int(lambda_exp)}_cutoff={t}_fips={c}.npy"
    tqdm.write("Saving {}".format(y_pred_fname))
    np.save(y_pred_fname, x_tc @ beta_tc.value)


    return True

# Define a helper function for parallel execution
def parallel_stage2_beta(args):
    stage1_beta_with_intercept, x_tc, y_tc, t, c, lambda_exp = args
    fname = f"./stage2_betas_organized/lambda_exp={int(lambda_exp)}/{t}/stage2_beta_lambda_exp={int(lambda_exp)}_cutoff={t}_fips={c}.npy"
    if os.path.exists(fname):
        tqdm.write(f"{fname} exists. Skipping")
        return None
    try:
        res = stage2_beta(*args)
        return res
    except Exception as e:
        # If an error occurs, log the error and return an error message or None
        tqdm.write(f"Error for t={t}, c={c}, lambda_exp={lambda_exp}: {str(e)}")
        return None


def task_generator(combined_data, t_values, c_values, lambda_exps):
    """
    Yields x_tc, y_tc, t, c, stage1_beta
    """
    for t in t_values:
        try:
            stage1_model = joblib.load("./stage1_lasso/best_lasso_model_cutoff={}.pkl".format(t))
            beta = stage1_model.coef_
            intercept = stage1_model.intercept_
            stage1_beta_with_intercept = np.concatenate((beta, [intercept]))
        
        except FileNotFoundError:
            tqdm.write(f"Model file not found for cutoff={t}. Skipping...")
            continue
        except Exception as e:
            tqdm.write(f"Error when loading model for cutoff={t}: {e}. Skipping...")
            continue
        for c in c_values:
            tqdm.write(f"Slicing (t={t}, c={c})")
            x_tc, y_tc = slice_data(combined_data, t, c)  # Perform slicing dynamically
            if x_tc.shape[0] == 0:
                tqdm.write(f"No data for t={t}, c={c}, Skipping")
                continue
            for lambda_exp in lambda_exps:
                yield (stage1_beta_with_intercept, x_tc, y_tc, t, c, lambda_exp)  # Yield each task as a tuple


def mp_parallelize_stage2_beta(combined_data=combined_data, t_values=list(range(50,1200)), c_values=fips_list, lambda_exps=lambda_exps):
    # Create a list of arguments (t, c, lambda) combinations
    results = []
    # Create a multiprocessing pool
    with Pool(processes=cpu_count()) as pool:
        # Submit tasks to the pool and use tqdm for progress bar
        #tasks = task_generator(combined_data, t_values, c_values, lambda_exps)
        
        # Use pool.imap to process tasks generated by the generator dynamically
        for result in tqdm(pool.imap(parallel_stage2_beta, task_generator(combined_data, t_values, c_values, lambda_exps)), total=len(t_values) * len(c_values) * len(lambda_exps)):
            results.append(result)  # Collect results
            
    return results


if __name__ == "__main__":

    start_cutoff = 50
    end_cutoff = 1200

    if len(sys.argv) != 3:
        print("Defaulting to start_cutoff := 50, end_cutoff := 1200")
    else:
        start_cutoff = int(sys.argv[1])
        end_cutoff = int(sys.argv[2])

    print(f"t_values = range({start_cutoff},{end_cutoff})")

    results = mp_parallelize_stage2_beta(combined_data=combined_data, t_values=list(range(start_cutoff, end_cutoff)), c_values=fips_list, lambda_exps=lambda_exps)


# In[ ]:




# In[ ]:




