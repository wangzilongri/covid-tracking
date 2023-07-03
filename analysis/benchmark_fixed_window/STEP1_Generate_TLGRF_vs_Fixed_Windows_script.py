#!/usr/bin/env python
# coding: utf-8

# In[19]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

import numpy as np
import pandas as pd
import ast
import glob
import pickle
import dask
import os
import itertools


#from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from statsmodels.regression.rolling import RollingOLS

#from tqdm.notebook import tqdm

from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
from tqdm import tqdm

import dask
import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import ProgressBar

#client = Client(n_workers=20, memory_limit="10GB", interface='lo')
from concurrent.futures import ThreadPoolExecutor

import dask_ml.cluster as dask_cluster

from pprint import pprint
import os

pd.set_option('display.max_columns', None)


# ### Generate Fixed Window Estimates of $wsize \in \{2,3,4,\dotsc,13,14\}$

# In[57]:


def RollingOLS_by_fips(df, fips, window_size, overwrite=False):
    directory = "./Fixed_Window_dfs"
    fname = "fixed_window_fips={}_wsize={}.csv".format(fips, window_size)
    os.makedirs(directory, exist_ok=True)
    csv_file = os.path.join(directory, fname)
    if not os.path.exists(csv_file) or overwrite:
        try:
            print("Generating betas for fips={}, window_size={}".format(fips, window_size))
            input_df = df.copy()
            input_df = input_df[input_df["fips"]==fips]
            input_df = input_df.sort_values(by="days_from_start")

            beta = RollingOLS(endog=input_df["log_rolled_cases"], exog=sm.add_constant(input_df["days_from_start"]), window=window_size).fit().params
            beta = beta.rename(columns={"days_from_start":"beta_wsize={}".format(window_size)})
            beta_df = pd.DataFrame(beta["beta_wsize={}".format(window_size)])
            beta_df["days_from_start"] = input_df["days_from_start"]
            beta_df["fips"] = fips
            beta_df = beta_df[["fips", "days_from_start","beta_wsize={}".format(window_size)]]
            #data_tuple = (fips, window_size, )
            beta_df.to_csv(csv_file, index=False)
            # Write the data to a pickle file
            #with open(pickle_file, "wb") as f:
            #    pickle.dump(beta_df, f)
        except:
            print("Something went wrong for fips={}, window_size={}".format(fips, window_size))
            return
    else:
        print("{} exists, skipping".format(csv_file))
        #with open(pickle_file, "rb") as f:
        #    beta_df = pickle.load(f)
    
    return


# In[ ]:

if __name__ == "__main__":

    augmented_df = dd.read_csv("../../data/augmented_us-counties_latest.csv", assume_missing=True).compute()
    augmented_df["date"] = pd.to_datetime(augmented_df["date"])
    augmented_df["fips"] = augmented_df["fips"].astype(int)
    augmented_df["days_from_start"] = augmented_df["days_from_start"].astype(int)
    augmented_df["log_rolled_cases"] = np.log(augmented_df["rolled_cases"] + 1.1)
    
    df = augmented_df.copy()
    window_sizes = range(2,15)
    #window_sizes = [3,4,5]
    fips_list = df["fips"].unique()
    
    #fips_list = fips_list[2000:]
    
    wsize_fips_list = list(itertools.product(fips_list, window_sizes))


    with tqdm(total=len(wsize_fips_list), desc="Processing") as pbar:
        def process_task(df, fips, window_size):
            # Perform the task
            result = RollingOLS_by_fips(df, fips, window_size)
            # Update the progress bar
            pbar.update(1)
            return result
        
        beta_results = Parallel(n_jobs=-1)(delayed(process_task)(df, fips, window_size) for fips, window_size in wsize_fips_list)
    pbar.close()


# In[ ]:




