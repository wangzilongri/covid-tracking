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


def RollingOLS_by_fips(df, fips, window_size):
    input_df = df.copy()
    input_df = input_df[input_df["fips"]==fips]
    input_df = input_df.sort_values(by="days_from_start")
    #display(df)
    print("Generating betas for fips={}, window_size={}".format(fips, window_size))
    beta = RollingOLS(endog=input_df["log_rolled_cases"], exog=sm.add_constant(input_df["days_from_start"]), window=window_size).fit().params
    beta = beta.rename(columns={"days_from_start":"beta_wsize={}".format(window_size)})
    return (fips, window_size, pd.DataFrame(beta["beta_wsize={}".format(window_size)]))


# In[ ]:

if __name__ == "__main__":

    benchmark_TLGRF_dataset = dd.read_csv("../benchmark_TLGRF_dataset.csv", assume_missing=True).compute()
    benchmark_TLGRF_dataset["date"] = pd.to_datetime(benchmark_TLGRF_dataset["date"])

    df = benchmark_TLGRF_dataset.copy()
    window_sizes = range(2,15)
    fips_list = df["fips"].unique()
    
    #window_sizes = [2,3,4]
    #fips_list = fips_list[:10]
    
    wsize_fips_list = list(itertools.product(fips_list, window_sizes))
    # In[79]:


    #window_sizes = [2,3,4]
    #fips_list = [1001,1003,99999]

    with tqdm(total=len(window_sizes) * len(fips_list), desc="Processing") as pbar:
        def update_progress(*args):
            pbar.update()

        beta_results = Parallel(n_jobs=-1)(delayed(RollingOLS_by_fips)(df, fips, window_size) for fips, window_size in wsize_fips_list)
        pbar.close()


    # In[92]:

    print("Appending results per window size")
    beta_result_dict = {window_size:[] for window_size in window_sizes}
    for result in beta_results:
        fips, window_size, beta_df = result
        beta_result_dict[window_size].append(beta_df)

    print("Concatenating results per window size")
    concatenated_beta_result_dict = {}
    for window_size, beta_df_list in beta_result_dict.items():
        concatenated_beta_result_dict[window_size] = pd.concat(beta_df_list)

    print("Merging each window size result")
    beta_df_big = pd.DataFrame()
    for window_size, big_beta_df in concatenated_beta_result_dict.items():
        beta_col_name = big_beta_df.columns[0]
        beta_df_big[beta_col_name] = big_beta_df

    print("Merging wih original DF")
    updated_df = pd.merge(df, beta_df_big, left_index=True, right_index=True, how="outer").sort_values(by=["fips", 'days_from_start'])

    print("Writing Output")
    updated_df.to_csv("TLGRF_w_Fixed_Windows.csv", index=False)


# In[ ]:




