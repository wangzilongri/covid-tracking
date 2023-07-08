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
import numpy as np
import pandas as pd
import statsmodels.api as sm

import dask
import dask.dataframe as dd
import dask_ml.cluster as dask_cluster
from dask.distributed import Client
from dask.diagnostics import ProgressBar

from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count

from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.cluster import KMeans

from statsmodels.regression.rolling import RollingOLS

from tqdm import tqdm
from collections import Counter
from functools import reduce
from pprint import pprint

pd.set_option('display.max_columns', None)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ### Generate/Load Merged Results

# In[101]:


### Params
args = sys.argv
script_name = args[0]  # The name of the script itself
K_start = int(args[1])  # The first argument
K_end = int(args[2]) # The end (inclusive)
K_step = int(args[3]) # Step
    
K_list = list(range(K_start, K_end + K_step, K_step))
#K_list = [1800]
### Directory to save/load merged best windows from
merged_directory = "./merged_kmeans_tcv_rolling_median"
os.makedirs(merged_directory, exist_ok=True)
### Load in the datasets
def merge_subroutine(K, REUSE_RESULTS = True):
    def read_csv_file(file_path):
        return dd.read_csv(file_path, assume_missing=True)
    kmeans_tcv_rolling_median_directory = "./kmeans_tcv_rolling_median"
    concatenated_dfs = {}
    K_subfolder = os.path.join(kmeans_tcv_rolling_median_directory,str(K))
    
    concatenated_df_fname = "merged_K={}.csv".format(K)
    concatenated_df_fpath = os.path.join(merged_directory, concatenated_df_fname)
    # Load file if exists, otherwise, create it and save it
    if os.path.exists(os.path.join(concatenated_df_fpath)) and REUSE_RESULTS:
        concatenated_df = pd.read_csv(concatenated_df_fpath)
        
    else:
        print("K={} does not exist! Creating".format(K))
        
        file_names = os.listdir(K_subfolder)
        file_paths = [os.path.join(K_subfolder, file_name) for file_name in file_names]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit the read_csv_file function to the executor for each file path
            dfs = list(tqdm(executor.map(read_csv_file, file_paths)))
        concatenated_df = dd.concat(dfs).compute()
        concatenated_df = concatenated_df.sort_values(by=["date", "k"])
        concatenated_df.to_csv(concatenated_df_fpath, index=False)
        #concatenated_dfs[K] = concatenated_df

    concatenated_df["date"] = pd.to_datetime(concatenated_df["date"])
    concatenated_df["date_query"] = concatenated_df["date"] + pd.Timedelta(days=7)
    concatenated_df = concatenated_df.drop(columns=["date"])
    
    return concatenated_df

concatenated_dfs = {}
for K in K_list:
    concatenated_dfs[K] = merge_subroutine(K)


# ### Load in Cluster Data

# In[3]:


kmeans_clusters_by_fips = pd.read_csv("../kmeans_clusters_by_fips.csv")
kmeans_clusters_by_fips


# ### Load in Diffs

# In[4]:


all_beta_df_results_diff = dd.read_csv("../Fixed_Windows_Validation_Diff.csv", assume_missing=True).compute()
all_beta_df_results_diff["date"] = pd.to_datetime(all_beta_df_results_diff["date"])
all_beta_df_results_diff = all_beta_df_results_diff.sort_values(by=["fips", "date"])

pattern = r'diff_wsize=\d+_shift=7'
filtered_cols = all_beta_df_results_diff.filter(regex=pattern).columns

# Include the filtered columns and desired non-matching columns
desired_cols = ['fips','date', 'days_from_start'] + list(filtered_cols)

all_beta_df_results_diff_test = all_beta_df_results_diff[desired_cols]
all_beta_df_results_diff_test["date"] = pd.to_datetime(all_beta_df_results_diff_test["date"])
all_beta_df_results_diff_test


# ### Obtain `fips_list` given cluster `Kk`

# In[5]:


def generate_mask(df=kmeans_clusters_by_fips, K=1, k=0):
    assert isinstance(K, int) and 1 <= K <= 3136, "K must be an integer $\in$ [1,3136]"
    assert isinstance(k, int) and 0 <= k < K, "k must be an integer $\in$ [0,k)"
    fips_list = df[df["kmeans_k={}_labels".format(K)] == k]["fips"].values
    
    return fips_list


# ### Set `date_query` to be 7 days after validation end

# In[102]:


###
REUSE_MERGED = True

#for K in tqdm(K_list):
def query_df_subroutine(K, concatenated_dfs=concatenated_dfs, REUSE_MERGED=True):
    generation_cols_to_convert = ["K","k", "best_mae_window", "best_rmse_window", "fips"]
    loading_cols_to_convert = ["K", "k", "fips"]
    
    tcv_kmeans_query_dfs_directory = "./kmeans_tcv_query_dfs"
    os.makedirs(tcv_kmeans_query_dfs_directory, exist_ok=True)

    
    query_df_fname = "query_K={}.csv".format(K)
    query_df_path = os.path.join(tcv_kmeans_query_dfs_directory, query_df_fname)
    if os.path.exists(os.path.join(query_df_path)) and REUSE_MERGED:
        query_df = pd.read_csv(query_df_path)
        query_df[loading_cols_to_convert] = query_df[loading_cols_to_convert].astype(np.int64)
        query_df["date_query"] = pd.to_datetime(query_df["date_query"])

    else:
        print("K={} does not exist! Generating".format(K))
        K_fips = kmeans_clusters_by_fips[["fips","kmeans_k={}_labels".format(K)]]
        query_df = pd.merge(concatenated_dfs[K], K_fips, left_on="k", right_on="kmeans_k={}_labels".format(K), how="left")
        query_df = query_df.sort_values(by=["date_query", "k", "fips"])


        query_df[generation_cols_to_convert] = query_df[generation_cols_to_convert].astype(np.int64)

        query_df["mae_diff_name"] = query_df['best_mae_window'].apply(lambda x: "diff_wsize={}_shift=7".format(x))
        query_df["rmse_diff_name"] = query_df['best_rmse_window'].apply(lambda x: "diff_wsize={}_shift=7".format(x))
        query_df = query_df.drop(columns=["kmeans_k={}_labels".format(K) , "best_mae_window", "best_rmse_window"])
        query_df.to_csv(query_df_path, index=False)
    return query_df
    
query_dfs = {}
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(query_df_subroutine, K_list), total=len(K_list)))

for K, result in zip(K_list, results):
    query_dfs[K] = result


# ### Merge with `all_beta_df_results_diff_test`

# In[81]:


daily_metrics_dfs = {}
#for K in tqdm(K_list):

#for K in tqdm(K_list):
def daily_metrics_subroutine(K, query_dfs=query_dfs, all_beta_df_results_diff_test=all_beta_df_results_diff_test):
    daily_metrics_dfs_directory = "./daily_metrics_dfs"
    os.makedirs(daily_metrics_dfs_directory, exist_ok=True)
    daily_metrics_fname = "daily_metrics_K={}.csv".format(K)
    daily_metrics_path = os.path.join(daily_metrics_dfs_directory, daily_metrics_fname)
    
    if os.path.exists(daily_metrics_path):
        daily_metrics_df = pd.read_csv(daily_metrics_path)
        daily_metrics_df["date_query"] = pd.to_datetime(daily_metrics_df["date_query"])
        daily_metrics_df.set_index('date_query', inplace=True)
        #daily_metrics_df.drop(columns='date_query', inplace=True)
    
    else:
        print("K={} does not exist! Creating...".format(K))
    
        query_diff_df = pd.merge(query_dfs[K], all_beta_df_results_diff_test, left_on=['date_query','fips'], right_on=['date','fips'])

        daily_metrics_df = pd.DataFrame()


        for val in ["mae", "rmse"]:
            dk_grouped = query_diff_df.groupby(["date_query","k"]).apply(lambda x : x[x["{}_diff_name".format(val)].unique()[0]].values)
            dk_grouped = pd.DataFrame(dk_grouped).rename(columns={0:"diffs"})
            dk_grouped_merged = dk_grouped.groupby('date_query')['diffs'].apply(lambda x: np.concatenate(x))

            for metric in ["MAE", "RMSE"]:
                cname = "Chosen_by_{}_{}".format(val, metric)
                if metric == "MAE":
                    daily_metrics_df[cname] = dk_grouped_merged.apply(lambda x : np.nanmean(np.abs(x)))
                else:
                    daily_metrics_df[cname] = dk_grouped_merged.apply(lambda x : np.sqrt(np.nanmean(np.square(x))))
        
        daily_metrics_df.to_csv(daily_metrics_path)
    return daily_metrics_df

daily_metrics_dfs = {}
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(daily_metrics_subroutine, K_list), total=len(K_list)))

for K, result in zip(K_list, results):
    daily_metrics_dfs[K] = result


# In[86]:


col_names = ["K"] + list(list(daily_metrics_dfs.values())[0].columns)
medians_df = pd.DataFrame(columns=col_names)
for K in K_list:
    row = [K] + list(daily_metrics_dfs[K].median())
    medians_df = medians_df.append(pd.Series(row, index=medians_df.columns), ignore_index=True)


# In[87]:


medians_df


# In[92]:


plt.figure(figsize=(20,10))
plt.scatter(medians_df["K"], medians_df["Chosen_by_mae_MAE"])
plt.xlabel("Number of Clusters: K")
plt.ylabel("Median MAE")
plt.title("Median MAE of kmeans + tcv predictor")
plt.show()


# In[91]:


plt.figure(figsize=(20,10))
plt.scatter(medians_df["K"], medians_df["Chosen_by_rmse_RMSE"])
plt.xlabel("Number of Clusters: K")
plt.ylabel("Median RMSE")
plt.title("Median RMSE of kmeans + tcv predictor")
plt.show()


# In[96]:


plt.figure(figsize=(20,10))
plt.plot(daily_metrics_dfs[1]["Chosen_by_mae_MAE"], label="K=1")
plt.plot(daily_metrics_dfs[3136]["Chosen_by_mae_MAE"], label="K=3136")
plt.xlabel("Number of Clusters: K")
plt.ylabel("Daily MAE")
plt.title("Daily MAE of kmeans + tcv predictor")
plt.legend()
plt.show()


# In[95]:


plt.figure(figsize=(20,10))
plt.plot(daily_metrics_dfs[1]["Chosen_by_rmse_RMSE"], label="K=1")
plt.plot(daily_metrics_dfs[3136]["Chosen_by_rmse_RMSE"], label="K=3136")
plt.xlabel("Number of Clusters: K")
plt.ylabel("Daily RMSE")
plt.title("Daily RMSE of kmeans + tcv predictor")
plt.legend()
plt.show()


# In[71]:


test_df.index = test_df["date_query"]


# In[72]:


test_df


# In[99]:


col_names = ["K"] + list(list(daily_metrics_dfs.values())[0].columns)
old_medians_df = pd.DataFrame(columns=col_names)
for K in K_list:
    old_daily_metrics = daily_metrics_dfs[K]
    old_daily_metrics = old_daily_metrics[old_daily_metrics.index >= "2020-03-06"]
    old_row = [K] + list(old_daily_metrics.median())
    old_medians_df = old_medians_df.append(pd.Series(old_row, index=medians_df.columns), ignore_index=True)


# In[100]:


old_medians_df


# In[ ]:




