#!/usr/bin/env python
# coding: utf-8

# ## Recovers tcv kmeans results and tries out both designs
# #### Design 1: Test time: Same `r` for all counties in a cluster
# #### Design 2: Test time: Separate `r` for all counties but same window used to estimate them

# In[78]:


import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

from collections import defaultdict
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed, load
from tqdm import tqdm

import ast
import glob
import pickle
import dask
import os
import itertools
import sys

import dask.dataframe as dd
from dask.distributed import Client

from pprint import pprint



def read_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def test_metric_worker(df, items, fips, t):
    X_cols = ["days_from_start"]
    y_col = "log_rolled_cases"
    
    best_rmse_windows = items[0]
    best_rmse_windowsize = best_rmse_windows[t]
    test_train_rmse = df[(t-best_rmse_windowsize <= df["days_from_start"]) & (df["days_from_start"] <= t)]
    test_data = df[df["days_from_start"] == t + 7]
    
    rmse_fips_df = test_train_rmse[test_train_rmse["fips"]==fips]
    test_fips_df = test_data[test_data["fips"]==fips]

    # Evaluate separately for each fips
    test_mse = 0
    nsamples_mse = 0
    lr_rmse = LinearRegression()
    if not rmse_fips_df.empty:
        try:
            lr_rmse.fit(rmse_fips_df[X_cols], rmse_fips_df[y_col])
            test_rmse_pred = lr_rmse.predict(test_fips_df[X_cols])
            test_mse = (mean_squared_error(test_rmse_pred, test_fips_df[y_col]))
            nsamples_mse = 1
            #test_mse_scores[t] += test_mse
            #test_mse_nsamples[t] += 1
        except:
            test_mse = 0
            nsamples_mse = 0
    
    best_mae_windows = items[1]
    best_mae_windowsize = best_mae_windows[t]
    test_train_mae = df[(t-best_mae_windowsize <= df["days_from_start"]) & (df["days_from_start"] <= t)]
    #test_data = df[df["days_from_start"] == t + 7]
    
    mae_fips_df = test_train_mae[test_train_mae["fips"]==fips]
    #test_fips_df = test_data[test_data["fips"]==fips]

    lr_mae = LinearRegression()
    test_mae = 0
    nsamples_mae = 0
    if not mae_fips_df.empty:
        try:
            lr_mae.fit(mae_fips_df[X_cols], mae_fips_df[y_col])
            test_mae_pred = lr_mae.predict(test_fips_df[X_cols])
            test_mae = mean_absolute_error(test_mae_pred, test_fips_df[y_col])
            nsamples_mae = 1
        except:
            test_mae = 0
            nsamples_mae = 0
    return np.asarray([t, test_mae, test_mse, nsamples_mse, nsamples_mae])


def test_time(hhs_clustered_panel_data, dfs_dict, cluster_key, max_window=14):
    """
    Feed in the dict items and parse the first 2
    Test time train a separate r on different counties in a cluster
    
    Return the total square error and absolute error
    """
    # Retrieve best windows at each time period
    K, k = cluster_key
    items = dfs_dict[cluster_key][cluster_key]
    #print(items)
    best_rmse_windows, best_mae_windows, best_rmse_test_design1, best_mae_test_design2 = items
    # Retrive the data
    df = hhs_clustered_panel_data[hhs_clustered_panel_data["kmeans_k={}_labels".format(K)] == k]
    df = df[["fips", "datetime", "days_from_start", "rolled_cases", "log_rolled_cases", "county", "state", "kmeans_k={}_labels".format(K)]]

    starting_day = hhs_clustered_panel_data["days_from_start"].min()
    # Save per day mae and rmse
    test_mse_scores = defaultdict(int)
    test_mae_scores = defaultdict(int)
    
    test_mse_nsamples = defaultdict(int)
    test_mae_nsamples = defaultdict(int)
    
    days = sorted(list(best_mae_windows.keys()))
    fips_list = sorted(list(df["fips"].unique()))
    
    fips_t_list = itertools.product(fips_list, days)
    
    print("Setting up parallel for cluster_key={}".format(cluster_key))
    with Parallel(n_jobs=-1, backend='multiprocessing') as parallel:
        metrics_samples_tuple_arr = parallel(delayed(test_metric_worker)(df, items, fips, t) for fips, t in tqdm(fips_t_list))    
    metrics_samples_tuple_matrix = np.asmatrix(metrics_samples_tuple_arr)
    print("Compiling scores and nsamples for cluster_key={}".format(cluster_key))
    #pprint(metrics_samples_tuple_matrix)
    
    column_names = ["days_from_start", "total_mse", "total_mae", "nsamples_mse", "nsamples_mae"]
    metrics_df = pd.DataFrame(metrics_samples_tuple_matrix, columns=column_names)
    
    metrics_df = metrics_df.groupby("days_from_start").sum()
    metrics_df = metrics_df.reset_index()
    
    return metrics_df


if __name__ == "__main__":
    
    K = int(sys.argv[1])
    
    # Load panel data
    hhs_clustered_panel_data = pd.read_csv("./hhs_clustered_panel_data.csv")
    # Generate list of desired Kk pairs (not all might be present)
    desired_file_names = set()
    # Retain a mapping from fnames to Kk
    fname_to_Kk_map = {}
    
    for k in range(K):
        desired_fname = os.path.join("kmeans_tcv_results", "cluster_tcv_dict_key=({},{}).pickle".format(K,k))
        desired_file_names.add(desired_fname)
        fname_to_Kk_map[desired_fname] = (K,k)
    
    available_files = set([os.path.join("kmeans_tcv_results", f) for f in os.listdir("kmeans_tcv_results") if f.endswith('.pickle')])
    
    pickle_files = desired_file_names.intersection(available_files)
    
    pickle_files = sorted(list(pickle_files))
    pool = Pool()
    cluster_key_list = [ast.literal_eval(s.split("=")[1].split(".")[0]) for s in pickle_files]
    dfs = pool.map(read_pickle, pickle_files)
    dfs_dict = dict(zip(cluster_key_list, dfs))
    
    available_Kk = sorted([fname_to_Kk_map[fname] for fname in pickle_files])
    
    os.makedirs("kmeans_tcv_2nd_design", exist_ok=True)
    for i, Kk in tqdm(enumerate(available_Kk)):
        K, k = Kk
        print("Generating 2nd design for {}, which is {}th out of {}".format(Kk, i+1, len(available_Kk)))
        metrics_df = test_time(hhs_clustered_panel_data, dfs_dict, Kk)
        metrics_df_fname = os.path.join("kmeans_tcv_2nd_design", "metrics_df_({},{}).csv".format(K,k))
        metrics_df.to_csv(metrics_df_fname, index=False)






