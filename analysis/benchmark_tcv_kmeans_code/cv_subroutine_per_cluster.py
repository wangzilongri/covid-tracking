#!/usr/bin/env python
# coding: utf-8

import ast
import concurrent.futures
import glob
import itertools
import os
import pickle
import sys
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

from tqdm import tqdm
from collections import Counter
from functools import reduce
from pprint import pprint

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def generate_mask(df, K, k):
    """
    Return a sorted list of fips conforming to a mask cluster
    """
    assert isinstance(K, int) and 1 <= K <= 3136, "K must be an integer $\in$ [1,3136]"
    assert isinstance(k, int) and 0 <= k < K, "k must be an integer $\in$ [0,k)"
    fips_list = df[df["kmeans_k={}_labels".format(K)] == k]["fips"].values
    
    return fips_list


def tcv_subroutine(df, fips_list, window_sizes=list(range(2,15)), shift_list=[7]):
    """
    Obtain the best window sizes by MAE and RMSE per date given a cluster mask (list of fips)
    """    
    df = df[df["fips"].isin(fips_list)]
    indexing_columns = ["days_from_start","date"]
    daily_metrics_df = df[indexing_columns].drop_duplicates()
    daily_metrics_df = daily_metrics_df.sort_values(by="days_from_start")
    daily_metrics_df = daily_metrics_df.reset_index(drop=True)

    for window_size in window_sizes:
        rmse = df.groupby("date").apply(lambda x : np.sqrt(np.nanmean(np.square(x[["diff_wsize={}_shift={}".format(window_size, s) for s in shift_list]]))))
        mae = df.groupby("date").apply(lambda x : np.nanmean(np.abs(x[["diff_wsize={}_shift={}".format(window_size, s) for s in shift_list]])))

        rmse = pd.DataFrame(rmse).reset_index().rename(columns={0:"rmse_wsize={}".format(window_size)})
        mae = pd.DataFrame(mae).reset_index().rename(columns={0:"mae_wsize={}".format(window_size)})

        daily_metrics_df = pd.merge(daily_metrics_df, mae, on="date", how="left")
        daily_metrics_df = pd.merge(daily_metrics_df, rmse, on="date", how="left")

        #daily_metrics_df["cumsum_mae_wsize={}".format(window_size)] = daily_metrics_df["mae_wsize={}".format(window_size)].cumsum()
        #daily_metrics_df["cumsum_rmse_wsize={}".format(window_size)] = daily_metrics_df["rmse_wsize={}".format(window_size)].cumsum()
    best_wsizes = pd.DataFrame(daily_metrics_df["date"])
    daily_mae_df = daily_metrics_df[["mae_wsize={}".format(window_size) for window_size in window_sizes]]
    best_wsizes["best_mae_window"] = daily_mae_df.idxmin(axis=1).str.split('=').str[1]
    
    daily_rmse_df = daily_metrics_df[["rmse_wsize={}".format(window_size) for window_size in window_sizes]]
    best_wsizes["best_rmse_window"] = daily_rmse_df.idxmin(axis=1).str.split('=').str[1]
    
    best_wsizes = best_wsizes[~best_wsizes["best_rmse_window"].isna()]
    best_wsizes = best_wsizes[~best_wsizes["best_mae_window"].isna()]
    
    best_wsizes["best_mae_window"] = best_wsizes["best_mae_window"].astype(np.int64)
    best_wsizes["best_rmse_window"] = best_wsizes["best_rmse_window"].astype(np.int64)
    
    best_wsizes = best_wsizes[best_wsizes["best_mae_window"] >= 2]
    best_wsizes = best_wsizes[best_wsizes["best_rmse_window"] >= 2]    
    
    return best_wsizes


def tcv_worker(all_beta_df_results_diff, kmeans_clusters_by_fips, Kk, window_sizes=list(range(2,15)), shift_list=[7], kmeans_tcv_validation_directory="./kmeans_tcv_validation"):
    K, k = Kk
    K_subfolder_directory = os.path.join(kmeans_tcv_validation_directory,str(K))
    os.makedirs(K_subfolder_directory, exist_ok=True)
    best_wsizes_directory = os.path.join(K_subfolder_directory, "best_wsizes_({},{}).csv".format(K,k))
    
    if os.path.exists(best_wsizes_directory):
        print("{} best wsizes exists! Skipping".format(Kk))
        return
    
    print("Executing tcv_worker on Cluster=({},{})".format(K,k))
    fips_list = generate_mask(kmeans_clusters_by_fips, K, k)
    best_wsizes = tcv_subroutine(all_beta_df_results_diff, fips_list, window_sizes=window_sizes, shift_list=shift_list)
    best_wsizes["K"] = K
    best_wsizes["k"] = k
    best_wsizes = best_wsizes[["K","k","date","best_mae_window", "best_rmse_window"]]
    best_wsizes.to_csv(best_wsizes_directory, index=False)
    
    return


if __name__ == "__main__":


    all_beta_df_results_diff = dd.read_csv("Fixed_Windows_Validation_Diff.csv", assume_missing=True).compute()
    all_beta_df_results_diff["date"] = pd.to_datetime(all_beta_df_results_diff["date"])
    all_beta_df_results_diff = all_beta_df_results_diff.sort_values(by=["fips", "date"])
    
    # Keep only diff_wsize={}_shift=7
    pattern = r'diff_wsize=\d+_shift=7'
    filtered_cols = all_beta_df_results_diff.filter(regex=pattern).columns
    # Include the filtered columns and desired non-matching columns
    desired_cols = ['fips','date', 'days_from_start'] + list(filtered_cols)
    
    all_beta_df_results_diff = all_beta_df_results_diff[desired_cols]
    all_beta_df_results_diff["date"] = pd.to_datetime(all_beta_df_results_diff["date"])
    all_beta_df_results_diff


    if os.path.exists("kmeans_clusters_by_fips.csv"):
        print("kmeans_clusters_by_fips.csv exists! Loading...")
        kmeans_clusters_by_fips = pd.read_csv("kmeans_clusters_by_fips.csv")
    else:
        print("kmeans_clusters_by_fips.csv does not exist! Generating...")
        hhs_X_w_clusters = pd.read_csv("hhs_X_w_clusters.csv")
        #hhs_X_w_clusters = hhs_X_w_clusters.filter(regex=r'^(fips|kmeans_k=\d+_labels)$')
        hhs_X_w_clusters["fips"] = hhs_X_w_clusters["fips"].astype(np.int64)
        hhs_X_w_clusters["kmeans_k=1_labels"] = 0
        hhs_X_w_clusters["kmeans_k=3136_labels"] = list(range(3136))
        # Get the kmeans_k={} part from column names
        column_names = hhs_X_w_clusters.columns[hhs_X_w_clusters.columns.str.contains(r'kmeans_k=\d+_labels')]
        # Sort the column names based on the numeric part
        sorted_column_names = sorted(column_names, key=lambda x: int(x.split('=')[1].split('_')[0]))
        kmeans_clusters_by_fips = hhs_X_w_clusters[['fips'] + sorted_column_names]
        kmeans_clusters_by_fips.to_csv("kmeans_clusters_by_fips.csv", index=False)

    
    args = sys.argv
    script_name = args[0]  # The name of the script itself
    K_start = int(args[1])  # The first argument
    K_end = int(args[2]) # The end (inclusive)
    K_step = int(args[3]) # Step
    
    K_list = list(range(K_start, K_end + K_step, K_step))
    
    kmeans_tcv_validation_directory = "./kmeans_tcv_validation"
    os.makedirs(kmeans_tcv_validation_directory, exist_ok=True)
    
    window_sizes = list(range(2,15))
    shift_list = [7]
    
    for K in K_list:
        Kk_list = [(K,k) for k in range(K)]
        print("Executing {} for {}".format(script_name, K))
        # Execute in Parallel
        with Parallel(n_jobs=-1, backend='multiprocessing') as parallel:
            best_wsizes_arr = parallel(delayed(tcv_worker)(all_beta_df_results_diff, kmeans_clusters_by_fips, Kk, window_sizes=window_sizes, shift_list=shift_list, kmeans_tcv_validation_directory=kmeans_tcv_validation_directory) for Kk in Kk_list)

