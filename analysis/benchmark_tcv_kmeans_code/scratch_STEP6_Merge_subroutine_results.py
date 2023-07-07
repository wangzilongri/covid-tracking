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

#from tqdm.notebook import tqdm
from tqdm import tqdm
from collections import Counter
from functools import reduce
from pprint import pprint

pd.set_option('display.max_columns', None)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ### Generate/Load Merged Results

# In[2]:


### Params
REUSE_RESULTS = True
K_list = list(range(1,21)) + list(range(100,3200,100)) + [3136]
#K_list = [1800]
### Directory to save/load merged best windows from
merged_directory = "./merged_kmeans_tcv_validation"
os.makedirs(merged_directory, exist_ok=True)
### Load in the datasets
kmeans_tcv_validation_directory = "./kmeans_tcv_validation"
concatenated_dfs = {}
for K in tqdm(K_list):
    K_subfolder = os.path.join(kmeans_tcv_validation_directory,str(K))
    
    concatenated_df_fname = "merged_K={}.csv".format(K)
    concatenated_df_fpath = os.path.join(merged_directory, concatenated_df_fname)
    # Load file if exists, otherwise, create it and save it
    if os.path.exists(os.path.join(concatenated_df_fpath)) and REUSE_RESULTS:
        concatenated_dfs[K] = pd.read_csv(concatenated_df_fpath)
        
    else:
        print("K={} does not exist! Creating".format(K))
        
        file_names = os.listdir(K_subfolder)
        file_paths = [os.path.join(K_subfolder, file_name) for file_name in file_names]
        dfs = [dd.read_csv(file, assume_missing=True) for file in file_paths]
        concatenated_df = dd.concat(dfs).compute()
        concatenated_df = concatenated_df.sort_values(by=["date", "k"])
        concatenated_df.to_csv(concatenated_df_fpath, index=False)
        concatenated_dfs[K] = concatenated_df

    concatenated_dfs[K]["date"] = pd.to_datetime(concatenated_dfs[K]["date"])
    concatenated_dfs[K]["date_query"] = concatenated_dfs[K]["date"] + pd.Timedelta(days=7)
    concatenated_dfs[K] = concatenated_dfs[K].drop(columns=["date"])


# ### Load in Cluster Data

# In[3]:


kmeans_clusters_by_fips = pd.read_csv("kmeans_clusters_by_fips.csv")
kmeans_clusters_by_fips


# ### Load in Diffs

# In[4]:


all_beta_df_results_diff = dd.read_csv("Fixed_Windows_Validation_Diff.csv", assume_missing=True).compute()
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

# In[6]:


###
REUSE_MERGED = True

query_dfs = {}
generation_cols_to_convert = ["K","k", "best_mae_window", "best_rmse_window", "fips"]
loading_cols_to_convert = ["K", "k", "fips"]


tcv_kmeans_query_dfs_directory = "./kmeans_tcv_query_dfs"
os.makedirs(tcv_kmeans_query_dfs_directory, exist_ok=True)

for K in tqdm(K_list):
    query_df_fname = "query_K={}.csv".format(K)
    query_df_path = os.path.join(tcv_kmeans_query_dfs_directory, query_df_fname)
    if os.path.exists(os.path.join(query_df_path)) and REUSE_MERGED:
        query_dfs[K] = pd.read_csv(query_df_path)
        query_dfs[K][loading_cols_to_convert] = query_dfs[K][loading_cols_to_convert].astype(np.int64)
        query_dfs[K]["date_query"] = pd.to_datetime(query_dfs[K]["date_query"])

    else:
        print("K={} does not exist! Generating".format(K))
        K_fips = kmeans_clusters_by_fips[["fips","kmeans_k={}_labels".format(K)]]
        query_dfs[K] = pd.merge(concatenated_dfs[K], K_fips, left_on="k", right_on="kmeans_k={}_labels".format(K), how="left")
        query_dfs[K] = query_dfs[K].sort_values(by=["date_query", "k", "fips"])


        query_dfs[K][generation_cols_to_convert] = query_dfs[K][generation_cols_to_convert].astype(np.int64)

        query_dfs[K]["mae_diff_name"] = query_dfs[K]['best_mae_window'].apply(lambda x: "diff_wsize={}_shift=7".format(x))
        query_dfs[K]["rmse_diff_name"] = query_dfs[K]['best_rmse_window'].apply(lambda x: "diff_wsize={}_shift=7".format(x))
        query_dfs[K] = query_dfs[K].drop(columns=["kmeans_k={}_labels".format(K) , "best_mae_window", "best_rmse_window"])
        query_dfs[K].to_csv(query_df_path, index=False)


# ### Merge with `all_beta_df_results_diff_test`

# In[ ]:


daily_metrics_dfs = {}
#for K in tqdm(K_list):
daily_metrics_dfs_directory = "./daily_metrics_dfs"
os.makedirs(daily_metrics_dfs_directory, exist_ok=True)

#daily_K_list = list(range(2500, 3200, 100))
#daily_K_list = [3136] + list(reversed(daily_K_list))
daily_K_list = [2000]
for K in tqdm(daily_K_list):
    daily_metrics_fname = "daily_metrics_K={}.csv".format(K)
    daily_metrics_path = os.path.join(daily_metrics_dfs_directory, daily_metrics_fname)
    
    if os.path.exists(daily_metrics_path):
        daily_metrics_df = pd.read_csv(daily_metrics_path)
        daily_metrics_df["date_query"] = pd.to_datetime(daily_metrics_df["date_query"])
        daily_metrics_df.set_index('date_query', inplace=True)
        daily_metrics_df.drop(columns='date_query', inplace=True)
    
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
    daily_metrics_dfs[K] = daily_metrics_df


# In[67]:


daily_metric_df


# In[68]:


daily_metric_df.median()


# In[74]:


K_list


# In[59]:


plt.plot(dk_grouped_merged)
plt.show()


# In[61]:


dk_grouped_merged


# In[69]:


test_df = daily_metric_df.reset_index()


# In[71]:


test_df.index = test_df["date_query"]


# In[72]:


test_df


# In[ ]:




