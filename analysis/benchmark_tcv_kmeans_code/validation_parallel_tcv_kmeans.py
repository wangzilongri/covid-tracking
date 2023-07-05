import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

from collections import defaultdict
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed, load
from tqdm import tqdm

import itertools
import os
import sys
import glob
import pickle
import dask
import dask.dataframe as dd
from dask.distributed import Client
#client = Client(n_workers=20, memory_limit="10GB", interface='lo')

from pprint import pprint


def calc_metrics(y_true, y_pred):
    #mse = np.sqrt(mean_squared_error(y_true, y_pred))
    #mae = mean_absolute_error(y_true, y_pred)
    se = np.square(y_true - y_pred)
    ae = np.abs(y_true - y_pred)
    
    return se, ae


def process_window(df, Kk, window_size, max_window=14, n_days_to_validate=14):
    print(f"Processing window_size {window_size}... for {Kk}")
    K, k = Kk
    
    starting_day = int(df["days_from_start"].min())
    
    X_cols = ["days_from_start"]
    y_col = "log_rolled_cases"
    # Initialize variables to store the MSE and MAE for each day of a window size
    se_scores = []
    ae_scores = []
    #Replace len(df["days_from_start"].unique()) with max
    range_object = range(max_window + starting_day, starting_day + df["days_from_start"].max()-n_days_to_validate-7)
    with tqdm(total=len(range_object)) as pbar:
        for i in range_object:
            train = df[(df["days_from_start"] >= i-window_size) & (df["days_from_start"]< i)]
            val = df[(i <= df["days_from_start"]) & (df["days_from_start"]< i + n_days_to_validate)]

            X_train = (train[X_cols])
            y_train = (train[y_col])
            X_val = (val[X_cols])
            y_val = (val[y_col])

            #print("For day {}, the dim of X_train={}, y_train={}".format(i, X_train.shape, y_train.shape))

            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_val)

            se, ae = calc_metrics(y_val, y_pred)

            se_scores.append(se)
            ae_scores.append(ae)
            pbar.update(1)
    all_results = Kk, np.cumsum(se_scores), np.cumsum(ae_scores)
    with open(os.path.join("./validation_kmeans_tcv_results/Kk_pickles", "validation_cluster_tcv_dict_key=({},{}).pickle".format(K,k)), 'wb') as f:
        print("Writing all_results for {}".format(Kk))
        pickle.dump(all_results, f)
    return all_results


def parallel_tcv(df, max_window=14, n_days_to_validate=14):
    # flatten df by adding all the log cases per day together
    #df = df.groupby('days_from_start')['log_rolled_cases'].sum().reset_index()
    #display(df)
    # Initialize variables to store overall validation mse and mae scores per window
    test_mse_scores = defaultdict(int)
    test_mae_scores = defaultdict(int)
    # Initialize variables to store the best window size and its corresponding MSE and MAE
    best_window_size_mse = {}
    best_window_size_mae = {}
    # Initialize variables to store the best mse and mae validation scores
    best_val_mse_scores = defaultdict(int)
    best_val_mae_scores = defaultdict(int)
    
    # Define the window size range
    window_size_range = range(2, max_window+1)
    X_cols = ['days_from_start']
    y_col = 'log_rolled_cases'
    
    n_rows =  len(df["days_from_start"].unique())-n_days_to_validate-7-max_window
    
    print(df)
    print("n_rows={}, len(df[days_from_start].unique()={}, n_days_to_validate={}, max_window={}".format(n_rows, len(df["days_from_start"].unique()), n_days_to_validate, max_window))
    
    val_mse_scores = np.zeros((max_window-1,n_rows))
    val_mae_scores = np.zeros((max_window-1, n_rows))
    
    starting_day = int(df["days_from_start"].min())
    #print(starting_day)
    
    with Parallel(n_jobs=-1, backend='multiprocessing') as parallel:
        score_tuple_arr = parallel(delayed(process_window)(df, window_size, max_window, n_days_to_validate) for window_size in (window_size_range))    
    for window_size in window_size_range:
        val_mse_scores[window_size-2,:] = score_tuple_arr[window_size-2][0]
        val_mae_scores[window_size-2,:] = score_tuple_arr[window_size-2][1]
    
    #print(np.shape(val_mse_scores))
    print("Getting best window size")
    for j, i in tqdm(enumerate(range(starting_day + max_window, starting_day + len(df["days_from_start"].unique())-n_days_to_validate-7))):        
        # Find the window size with the lowest MSE and MAE
        best_mse_windowsize = np.argmin(val_mse_scores[:,j]) + 2
        best_mae_windowsize = np.argmin(val_mae_scores[:,j]) + 2
        
        #print("For day {}, the best window size for mse={}, mae={}".format(i, best_mse_windowsize, best_mae_windowsize))

        best_window_size_mse[i] = best_mse_windowsize 
        best_window_size_mae[i] = best_mae_windowsize 

        best_val_mse_scores[i] = val_mse_scores[best_mse_windowsize - 2, j]
        best_val_mae_scores[i] = val_mae_scores[best_mae_windowsize - 2, j]

        # Get the test error
        test_train_mse = df[(i-best_mse_windowsize <= df["days_from_start"]) & (df["days_from_start"] < i)]
        test_train_mae = df[(i-best_mae_windowsize <= df["days_from_start"]) & (df["days_from_start"] < i)]
        
        test_data = df[(df["days_from_start"] == i + 6)]

        lr_mse = LinearRegression()
        lr_mse.fit(test_train_mse[X_cols], test_train_mse[y_col])
        test_mse_pred = lr_mse.predict(test_data[X_cols])
        test_mse = np.nanmean(np.square(test_mse_pred, test_data[y_col]))

        lr_mae = LinearRegression()
        lr_mae.fit(test_train_mae[X_cols], test_train_mae[y_col])
        test_mae_pred = lr_mae.predict(test_data[X_cols])
        test_mae = np.nanmean(np.abs(test_mae_pred, test_data[y_col]))

        test_mse_scores[i] = test_mse
        test_mae_scores[i] = test_mae
    return best_window_size_mse, best_window_size_mae, test_mse_scores, test_mae_scores



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
        mask_dict[(K,k)] = mask
    return mask_dict



def parallel_process_window(df, K, max_window=14, n_days_to_validate=14):
    mask_dict = generate_mask(df, K)
    cluster_tcv_dict = {}
    window_size_range = list(range(2, max_window))
    
    Kk_list = sorted(list(cluster_tcv_dict.keys()))
    
    Kk_window_product = itertools.product(Kk_list, window_size_range)
    os.makedirs("./validation_kmeans_tcv_results/Kk_pickles", exist_ok=True)
    
    print("Generating parallel process_window workers for {}".format(K))
    
    score_tuple_arr_dict = {}
    for Kk, window_size in (Kk_window_product):
        score_tuple_arr_dict[Kk] = process_window(df[mask_dict[Kk]], Kk, window_size, max_window, n_days_to_validate)
    
    #with Parallel(n_jobs=-1, backend='multiprocessing') as parallel:
    #    score_tuple_arr = parallel(delayed(process_window)(df[mask_dict[Kk]], Kk, window_size, max_window, n_days_to_validate) for Kk, window_size in (Kk_window_product))
    return score_tuple_arr_dict


if __name__ == "__main__":
    
    print("Starting validation_parallel_tcv_kmeans.py")
    
    print("Reading in hhs_clustered_panel_data.csv")
    df = dd.read_csv("./hhs_clustered_panel_data.csv", assume_missing=True).compute()
    mask = np.logical_not(np.isfinite(df['log_rolled_cases']))
    df = df[~mask]
    #df = hhs_clustered_panel_data
    
    # Conduct clustered tcv
    K = int(sys.argv[1])
    #k = int(sys.argv[2])
    
    max_window = 14
    
    print("Starting validation for ({})".format(K))
    os.makedirs("./validation_kmeans_tcv_results", exist_ok=True)
    os.makedirs("./validation_kmeans_tcv_results/K_pickles", exist_ok=True)
    cluster_tcv_dict = parallel_process_window(df, K, max_window=14, n_days_to_validate=14)
    
    
    with open(os.path.join("./validation_kmeans_tcv_results/K_pickles","validation_cluster_tcv_dict_key={}.pickle".format(K)), 'wb') as f:
        pickle.dump(cluster_tcv_dict, f)