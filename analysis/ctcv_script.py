import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

from collections import defaultdict
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed, load
from tqdm import tqdm

import os
import sys
import glob
import pickle
import dask
import dask.dataframe as dd
from dask.distributed import Client
#client = Client(n_workers=20, memory_limit="10GB", interface='lo')

from pprint import pprint


hhs_clustered_panel_data = dd.read_csv("./hhs_clustered_panel_data.csv", assume_missing=True).compute()
mask = np.logical_not(np.isfinite(hhs_clustered_panel_data['log_rolled_cases']))
hhs_clustered_panel_data = hhs_clustered_panel_data[~mask]

def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def process_window(df, starting_day, window_size, max_window=14, n_days_to_validate=2):
    print(f"Processing window_size {window_size}...")
    X_cols = ["days_from_start"]
    y_col = "log_rolled_cases"
    # Initialize variables to store the RMSE and MAE for each day of a window size
    rmse_scores = []
    mae_scores = []
    range_object = range(max_window + starting_day, starting_day + len(df["days_from_start"].unique())-n_days_to_validate-7)
    with tqdm(total=len(range_object)) as pbar:
        for i in range_object:
            train = df[ (df["days_from_start"] >= i-window_size) &  (df["days_from_start"]< i)]
            val = df[ (i <= df["days_from_start"]) & (df["days_from_start"]< i + n_days_to_validate)]

            X_train = (train[X_cols])
            y_train = (train[y_col])
            X_val = (val[X_cols])
            y_val = (val[y_col])

            #print("For day {}, the dim of X_train={}, y_train={}".format(i, X_train.shape, y_train.shape))

            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_val)

            rmse, mae = calc_metrics(y_val, y_pred)

            rmse_scores.append(rmse)
            mae_scores.append(mae)
            pbar.update(1)
    return np.cumsum(rmse_scores), np.cumsum(mae_scores)


def parallel_tcv(df, max_window=14, n_days_to_validate=2):
    # flatten df by adding all the log cases per day together
    #df = df.groupby('days_from_start')['log_rolled_cases'].sum().reset_index()
    #display(df)
    # Initialize variables to store overall validation rmse and mae scores per window
    test_rmse_scores = defaultdict(int)
    test_mae_scores = defaultdict(int)
    # Initialize variables to store the best window size and its corresponding RMSE and MAE
    best_window_size_rmse = {}
    best_window_size_mae = {}
    # Initialize variables to store the best rmse and mae validation scores
    best_val_rmse_scores = defaultdict(int)
    best_val_mae_scores = defaultdict(int)
    
    # Define the window size range
    window_size_range = range(2, max_window+1)
    X_cols = ['days_from_start']
    y_col = 'log_rolled_cases'
    
    n_rows =  len(df["days_from_start"].unique())-n_days_to_validate-7-max_window
    
    val_rmse_scores = np.zeros((max_window-1,n_rows))
    val_mae_scores = np.zeros((max_window-1, n_rows))
    
    starting_day = int(df["days_from_start"].min())
    #print(starting_day)
    
    with Parallel(n_jobs=-1, backend='multiprocessing') as parallel:
        score_tuple_arr = parallel(delayed(process_window)(df, starting_day, window_size, max_window, n_days_to_validate) for window_size in (window_size_range))    
    for window_size in window_size_range:
        val_rmse_scores[window_size-2,:] = score_tuple_arr[window_size-2][0]
        val_mae_scores[window_size-2,:] = score_tuple_arr[window_size-2][1]
    
    #print(np.shape(val_rmse_scores))
    print("Getting best window size")
    for j, i in tqdm(enumerate(range(starting_day + max_window, starting_day + len(df["days_from_start"].unique())-n_days_to_validate-7))):        
        # Find the window size with the lowest RMSE and MAE
        best_rmse_windowsize = np.argmin(val_rmse_scores[:,j]) + 2
        best_mae_windowsize = np.argmin(val_mae_scores[:,j]) + 2
        
        #print("For day {}, the best window size for rmse={}, mae={}".format(i, best_rmse_windowsize, best_mae_windowsize))

        best_window_size_rmse[i] = best_rmse_windowsize 
        best_window_size_mae[i] = best_mae_windowsize 

        best_val_rmse_scores[i] = val_rmse_scores[best_rmse_windowsize - 2, j]
        best_val_mae_scores[i] = val_mae_scores[best_mae_windowsize - 2, j]

        # Get the test error
        test_train_rmse = df[(i-best_rmse_windowsize <= df["days_from_start"]) & (df["days_from_start"] < i)]
        test_train_mae = df[(i-best_mae_windowsize <= df["days_from_start"]) & (df["days_from_start"] < i)]
        
        test_data = df[(df["days_from_start"] == i + 6)]
        
        #display(test_data)
        
        #print("For day {}, the dim of test_train_rmse_X={}, test_train_rmse_y={}".format(i, test_train_rmse[X_cols].shape,test_train_rmse[y_col].shape))
        #print("For day {}, the dim of test_train_mae_X={}, test_train_mae_y={}".format(i, test_train_mae[X_cols].shape,test_train_mae[y_col].shape))

        lr_rmse = LinearRegression()
        lr_rmse.fit(test_train_rmse[X_cols], test_train_rmse[y_col])
        test_rmse_pred = lr_rmse.predict(test_data[X_cols])
        test_rmse = np.sqrt(mean_squared_error(test_rmse_pred, test_data[y_col]))

        lr_mae = LinearRegression()
        lr_mae.fit(test_train_mae[X_cols], test_train_mae[y_col])
        test_mae_pred = lr_mae.predict(test_data[X_cols])
        test_mae = mean_absolute_error(test_mae_pred, test_data[y_col])

        test_rmse_scores[i] = test_rmse
        test_mae_scores[i] = test_mae
    return best_window_size_rmse, best_window_size_mae, test_rmse_scores, test_mae_scores


def feed_parallel_tcv(mask, df):
    return parallel_tcv(df[mask], max_window=14, n_days_to_validate=2)


def generate_masks(df):
    # kmeans_string
    # e.g. kmeans_k=2_labels
    mask_dict = {}
    for K in range(2, 21):
        kmeans_string = "kmeans_k={}_labels".format(K)
        for k in range(K):
            mask = (df[kmeans_string]==k)
            n = np.sum(mask)
            mask_dict[(K,k)] = (mask, n)
    return mask_dict

def generate_mask(df, K):
    kmeans_string = "kmeans_k={}_labels".format(K)
    mask_dict = {}
    for k in range(K):
        mask = (df[kmeans_string]==k)
        n = np.sum(mask)
        mask_dict[(K,k)] = (mask, n)
    return mask_dict

def serial_clustered_tcv(df, K):
    mask_dict = generate_mask(df, K)
    #with Parallel(n_jobs=-1, backend='multiprocessing') as parallel:
        #overall_results_arr = parallel(delayed(feed_parallel_tcv)(cluster_item[1][0], df) for (cluster_item) in mask_dict.items())
    cluster_tcv_dict = {}
    for i, cluster_item in tqdm(enumerate(mask_dict.items())):
        cluster_key, cluster_mask_tuple = cluster_item
        print("Handling cluster={}".format(cluster_key))
        cluster_mask, n = cluster_mask_tuple
        cluster_tcv_dict[cluster_key] = feed_parallel_tcv(cluster_mask, df)
    return cluster_tcv_dict

def serial_clustered_tcv_by_Kk(df, K, k):
    kmeans_string = "kmeans_k={}_labels".format(K)
    mask = (df[kmeans_string]==k)
    cluster_tcv_dict = {}
    print("Handling cluster={}".format((K,k)))
    cluster_tcv_dict[(K,k)] = feed_parallel_tcv(mask, df)
    return cluster_tcv_dict




#test_df = hhs_clustered_panel_data[hhs_clustered_panel_data["days_from_start"] <= 150]
test_df = hhs_clustered_panel_data

# ctcv procedure
if True:
    ctcv_compiled_dict = {}
    os.makedirs("./ctcv_by_fips", exist_ok=True)

    fips_list = sorted(test_df["fips"].unique())

    starting = int(sys.argv[1])
    ending = int(sys.argv[2])

    print("Starting = {}, ending={}".format(starting, ending))

    fips_list = fips_list[starting:ending]

    for fips in tqdm(fips_list):
        fips_mask = test_df["fips"]==fips
        fname = os.path.join("./ctcv_by_fips","ctcv_fips={}.pickle".format(int(fips)))
        if os.path.exists(fname):
            print("{} exists, moving on".format(fname))
            continue
        try:
            results_tuple = feed_parallel_tcv(fips_mask, test_df)
            ctcv_compiled_dict["fips"] = results_tuple
            with open(fname, "wb") as f:
                pickle.dump(results_tuple, f)
        except:
            continue

