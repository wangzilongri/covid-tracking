#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

from collections import defaultdict
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed, load
#from tqdm.autonotebook import tqdm
from tqdm import tqdm

import ast
import glob
import pickle
import dask
import os
import dask.dataframe as dd
from dask.distributed import Client
#client = Client(n_workers=20, memory_limit="10GB", interface='lo')

from pprint import pprint


# ### Load kmeans data and classifiers

# In[2]:


#augmented_data_path = "../data/augmented_us-counties-states_latest.csv"
#augmented_df = (dd.read_csv(augmented_data_path, assume_missing=True))
# feature data
#hhs_X_path = "./hhs_kmeans_data.csv"
#hhs_X_df = dd.read_csv(hhs_X_path, assume_missing=True).compute()
#X = hhs_X_df.drop(["fips", "county", "state"], axis=1)
#imp = SimpleImputer(strategy='mean')
#X = imp.fit_transform(X)
#hhs_X_df.head()
#hhs_y_df = augmented_df[["fips","datetime","days_from_start","rolled_cases"]].compute().sort_values(["fips","days_from_start"])
#hhs_y_df["log_rolled_cases"] = np.log(hhs_y_df["rolled_cases"] + 1e-10)


# In[3]:


#kmeans_classifiers_folder_path = "./kmeans_classifiers"
#def load_kmeans(fname):
#    return load(fname)
#kmeans_classifiers_paths = glob.glob("./kmeans_classifiers/kmeans_*.joblib")
#K_list = [int(s.split("/")[2].split("_")[1].split(".")[0]) for s in kmeans_classifiers_paths]
#K_list = sorted(K_list)
#kmeans_classifiers = Parallel(n_jobs=-1)(delayed(load_kmeans)(f) for f in sorted(kmeans_classifiers_paths))


# In[4]:


#K_list


# In[ ]:


#(kmeans_classifiers)[-2].cluster_centers_


# In[6]:


def predict_label(kmeans_classifier):
    k_clusters = kmeans_classifier.get_params()["n_clusters"]
    labels = kmeans_classifier.predict(X).compute()
    label_col = pd.DataFrame({"kmeans_k={}_labels".format(k_clusters): labels})
    return label_col
#labels_arr = Parallel(n_jobs=-1)(delayed(predict_label)(c) for c in (kmeans_classifiers))


# In[7]:


#hhs_X_w_clusters_df = pd.concat([hhs_X_df, pd.concat(labels_arr, axis=1)], axis=1)
#hhs_X_w_clusters_df.to_csv("./hhs_X_w_clusters.csv", index=False)


# In[8]:


#hhs_labels_df = hhs_X_w_clusters_df[["fips", "county","state"] + ["kmeans_k={}_labels".format(k) for k in K_list]]
#hhs_labels_df.head()


# In[9]:


#n_fips = hhs_labels_df.shape[0]
#hhs_labels_df["kmeans_k={}_labels".format(n_fips)] = hhs_labels_df.index
#hhs_labels_df["kmeans_k={}_labels".format(1)] = 0
#hhs_labels_df


# In[10]:


#hhs_clustered_panel_data = pd.merge(hhs_y_df, hhs_labels_df, how="inner", on="fips")
#hhs_clustered_panel_data["log_rolled_cases"] = hhs_clustered_panel_data["log_rolled_cases"].apply(lambda x: max(0,x))
#print(hhs_clustered_panel_data.shape)
#(hhs_clustered_panel_data.head())
#hhs_clustered_panel_data.to_csv("./hhs_clustered_panel_data.csv", index=False)
hhs_clustered_panel_data = pd.read_csv("./hhs_clustered_panel_data.csv")

# In[11]:


#mask = np.logical_not(np.isfinite(hhs_clustered_panel_data['log_rolled_cases']))
#hhs_clustered_panel_data = hhs_clustered_panel_data[~mask]


# In[ ]:





# In[12]:


# Define a function to calculate RMSE and MAE
def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


# In[13]:


#np.log


# In[14]:


# Feeds in mask to generate slice

def tcv(df, max_window=14, n_days_to_validate=2):
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
    print(starting_day)
    for window_size in window_size_range:
        print(f"Processing window_size {window_size}...")
        # Initialize variables to store the RMSE and MAE for each day of a window size
        rmse_scores = []
        mae_scores = []
        for i in tqdm(range(max_window + starting_day, starting_day + len(df["days_from_start"].unique())-n_days_to_validate-7)):
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
        val_rmse_scores[window_size-2,:] = np.cumsum(rmse_scores)
        val_mae_scores[window_size-2,:] = np.cumsum(mae_scores)
    
    print(np.shape(val_rmse_scores))
    print("Getting best window size")
    for j, i in tqdm(enumerate(range(starting_day + max_window, starting_day + len(df["days_from_start"].unique())-n_days_to_validate-7))):        
        # Find the window size with the lowest RMSE and MAE
        best_rmse_windowsize = np.argmin(val_rmse_scores[:,j]) + 2
        best_mae_windowsize = np.argmin(val_mae_scores[:,j]) + 2
        
        print("For day {}, the best window size for rmse={}, mae={}".format(i, best_rmse_windowsize, best_mae_windowsize))

        best_window_size_rmse[i] = best_rmse_windowsize 
        best_window_size_mae[i] = best_mae_windowsize 

        best_val_rmse_scores[i] = val_rmse_scores[best_rmse_windowsize - 2, j]
        best_val_mae_scores[i] = val_mae_scores[best_mae_windowsize - 2, j]

        # Get the test error
        test_train_rmse = df[(i-best_rmse_windowsize <= df["days_from_start"]) & (df["days_from_start"] < i)]
        test_train_mae = df[(i-best_mae_windowsize <= df["days_from_start"]) & (df["days_from_start"] < i)]
        
        test_data = df[(i <= df["days_from_start"]) & (df["days_from_start"] < i + 7)]
        
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

def feed_tcv(mask):
    df = hhs_clustered_panel_data[mask]
    return tcv(df) 


# In[15]:


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
        score_tuple_arr = parallel(delayed(process_window)(df, starting_day, window_size, max_window, n_days_to_validate) for window_size in tqdm(window_size_range))    
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
        
        test_data = df[(i <= df["days_from_start"]) & (df["days_from_start"] < i + 7)]
        
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


# In[16]:


def generate_masks(K_list):
    # kmeans_string
    # e.g. kmeans_k=2_labels
    mask_dict = {}
    for K in K_list:
        kmeans_string = "kmeans_k={}_labels".format(K)
        for k in range(K):
            mask = (hhs_clustered_panel_data[kmeans_string]==k)
            n = np.sum(mask)
            mask_dict[(K,k)] = (mask, n)
    return mask_dict


# In[17]:


def generate_mask(K):
    # kmeans_string
    # e.g. kmeans_k=2_labels
    mask_dict = {}
    kmeans_string = "kmeans_k={}_labels".format(K)
    for k in range(K):
        mask = (hhs_clustered_panel_data[kmeans_string]==k)
        n = np.sum(mask)
        mask_dict[(K,k)] = (mask, n)
    return mask_dict


# In[18]:


#K_list = list(range(1,21)) + list(range(100,3100,100)) + [3136]
#generate_mask(600)


# ### Load all the Pickle Results

# In[ ]:


def read_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

    
# In[ ]:


#dfs_dict[(600,531)]


# In[ ]:


def recompute_metrics(results_dict):
    
    cluster_key = list(results_dict.keys())[0]
    mask = generate_mask(cluster_key[0])[cluster_key][0]
    df = hhs_clustered_panel_data[mask]
    n_samples_by_day = df.groupby("days_from_start")["rolled_cases"].count()
    
    results_dicts_tuple = list(results_dict.values())[0]
    
    test_mse_dict = results_dicts_tuple[2]
    test_mae_dict = results_dicts_tuple[3]
    
    #starting = int(min(test_rmse_dict.keys()))
    #ending = int(max(test_rmse_dict.keys()))
    
    compiled_dict = {}
    for day in test_mae_dict.keys():
        se = test_mse_dict[day] * n_samples_by_day[day]
        ae = test_mae_dict[day] * n_samples_by_day[day]
        count = n_samples_by_day[day]
        
        compiled_dict[day] = (se, ae, count)
    
    compiled_df = pd.DataFrame.from_dict(compiled_dict, orient="index", columns=['se', 'ae', 'count'])
    return compiled_df


# In[ ]:


def recompute_cluster_metrics(K, dfs_dict):
    Kk_list = [(K,k) for k in range(K)]
    results_dict_list = []
    for Kk in Kk_list:
        if Kk in dfs_dict.keys():
            results_dict_list.append(dfs_dict[Kk])
    with Parallel(n_jobs=-1, backend='multiprocessing') as parallel:
        compiled_df_list = parallel(delayed(recompute_metrics)(results_dict) for results_dict in (results_dict_list))
    merged_result = pd.concat(compiled_df_list, axis=0)
    merged_result = merged_result.groupby(merged_result.index).sum()
    
    merged_result["rmse"] = np.sqrt(merged_result["se"]/merged_result["count"])
    merged_result["mae"] = (merged_result["ae"]/merged_result["count"])
    
    return merged_result[["rmse","mae"]]


def serial_recompute_cluster_metrics(K, dfs_dict):
    Kk_list = [(K,k) for k in range(K)]
    results_dict_list = []

    merged_results_list = []
    for Kk in Kk_list:
        K,k = Kk
        if Kk in dfs_dict.keys():
            print("Computing (K={},k={})".format(K,k))
            merged_results_list.append(recompute_metrics(dfs_dict[Kk]))
            
    #merged_result = pd.DataFrame()

    merged_result = pd.concat(merged_results_list, axis=0)
    merged_result = merged_result.groupby(merged_result.index).sum()
    
    merged_result["rmse"] = np.sqrt(merged_result["se"]/merged_result["count"])
    merged_result["mae"] = (merged_result["ae"]/merged_result["count"])
    
    return merged_result[["rmse","mae"]]

# In[ ]:


#merged_result = recompute_cluster_metrics(2)


# In[ ]:


#merged_result


# In[ ]:


K_list = list(range(1,21)) + list(range(100,3200,100)) + [3136]
K_list = list(range(3000, 3200, 100)) + [3136]
K_list = [3100]
# In[ ]:

pickle_files = [os.path.join("kmeans_tcv_results", f) for f in os.listdir("kmeans_tcv_results") if f.endswith('.pickle')]
pickle_files = sorted(pickle_files)
pool = Pool()

cluster_key_list = [ast.literal_eval(s.split("=")[1].split(".")[0]) for s in pickle_files]

print("Reading all pickle files")
dfs = pool.map(read_pickle, pickle_files)

dfs_dict = dict(zip(cluster_key_list, dfs))


merged_result_dict = {}
kmeans_tcv_merged_results_dir = "kmeans_tcv_merged_results"
os.makedirs(kmeans_tcv_merged_results_dir, exist_ok=True)
for K in tqdm(K_list):
    print("Recomputing cluster={}".format(K))
    #merged_result_dict[K] = recompute_cluster_metrics(K, dfs_dict)

    merged_result_dict[K] = serial_recompute_cluster_metrics(K, dfs_dict)
    with open(os.path.join(kmeans_tcv_merged_results_dir, "kmeans_tcv_merged_K={}.pickle".format(K)), "wb") as f:
        pickle.dump(merged_result_dict[K], f)
    #except:
    #    print("Something went wrong with K={}".format(K))


# In[ ]:


#with open("./cluster_tcv_dict_key=(1,0).pickle", "rb") as f:
#    tcv_dict = pickle.load(f)
#tcv_results = pd.DataFrame.from_dict(tcv_dict[2], orient='index', columns=["rmse"])
#tcv_results["mae"] = tcv_dict[3].values()
#tcv_results["rmse"]
#tcv_results["mae"]
#merged_result_dict[1] = tcv_results
#merged_result_dict


# In[ ]:


#with open("./cluster_tcv_merged_result_dict.pickle", "wb") as f:
#    pickle.dump(merged_result_dict, f)


# In[ ]:





# In[ ]:


#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot the daily RMSEs for each dataframe
figsize=(20,5)
plt.subplots(figsize=figsize)

for k, df in merged_result_dict.items():
    plt.plot(df.index, df['rmse'], label='RMSE of #Clusters={}'.format(k))

# Add a legend to the first subplot
plt.legend()
plt.title('Daily RMSEs')
plt.xlabel('Day')
plt.ylabel('RMSE')

plt.show()
plt.subplots(figsize=figsize)
for k, df in merged_result_dict.items():
    plt.plot(df.index, df['mae'], label='MAE of #Clusters={}'.format(k))

# Add a legend to the first subplot
plt.legend()
plt.title('Daily MAEs')
plt.xlabel('Day')
plt.ylabel('MAE')

plt.show()


# In[ ]:


median_dict = {k: df.median() for k, df in merged_result_dict.items()}
median_df = pd.DataFrame(median_dict).transpose()

# Add a column indicating the value of K
median_df.index.name = 'K'
median_df.reset_index(inplace=True)

# Rename the columns
median_df = median_df.rename(columns={'rmse': 'Median RMSE', 'mae': 'Median MAE'})
median_df = median_df.sort_values(by=["K"])

lowest_rmse_K_df = median_df.sort_values(by=["Median RMSE"]).iloc[0]
print("Cluster={} has lowest Median RMSE of {}, with median MAE of {}".format(lowest_rmse_K_df["K"], lowest_rmse_K_df["Median RMSE"], lowest_rmse_K_df["Median MAE"]))
lowest_mae_K_df = median_df.sort_values(by=["Median MAE"]).iloc[0]
print("Cluster={} has lowest Median MAE of {}, with median RMSE of {}".format(lowest_mae_K_df["K"], lowest_mae_K_df["Median MAE"], lowest_mae_K_df["Median RMSE"]))

tcv_df = median_df.loc[median_df["K"]==1, ["Median RMSE", "Median MAE"]].values
print("tcv i.e. Cluster=1 has Median MAE={}, Median RMSE={}".format(tcv_df[0][0], tcv_df[0][1]))
ctcv_df = median_df.loc[median_df["K"]==3136, ["Median RMSE", "Median MAE"]].values
print("ctcv i.e. Cluster=3136 has Median MAE={}, Median RMSE={}".format(median_df[median_df["K"]==3136].iloc[0]["Median MAE"], median_df[median_df["K"]==3136].iloc[0]["Median RMSE"]))
median_df.to_csv("kmeans_tcv_merged_median_metrics.csv", index=False)
median_df


# In[ ]:


median_df.to_csv("temp_merged_cluster_tcv.csv", index=False)


# ### Fixed Window Sizes

# In[ ]:


def fixed_window_worker(df, window_size):
    X_cols = ['days_from_start']
    y_col = 'log_rolled_cases'
    starting_day = int(df["days_from_start"].min())
    print(f"Processing window_size {window_size}...")
    # Initialize variables to store the RMSE and MAE for each day of a window size
    scores = {}
    for i in tqdm(range(window_size + starting_day, starting_day + len(df["days_from_start"].unique())-7)):
        train = df[ (df["days_from_start"] >= i-window_size) &  (df["days_from_start"]< i)]
        test = df[df["days_from_start"]== i + 6]

        X_train = (train[X_cols])
        y_train = (train[y_col])
        X_test = (test[X_cols])
        y_test = (test[y_col])

        #print("For day {}, the dim of X_train={}, y_train={}".format(i, X_train.shape, y_train.shape))

        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse, mae = calc_metrics(y_test, y_pred)

        scores[i] = (rmse, mae)
    return scores

def fixed_window_parallel(df, max_window=14):
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
    
    n_rows =  len(df["days_from_start"].unique())-max_window
    
    starting_day = int(df["days_from_start"].min())
    print(starting_day)
    with Parallel(n_jobs=-1, backend='multiprocessing') as parallel:
        results_arr = parallel(delayed(fixed_window_worker)(df, window_size) for window_size in (window_size_range))
        
            
    return results_arr


# In[ ]:





# In[ ]:


# fixed_windows_results_arr = fixed_window_parallel(hhs_clustered_panel_data, max_window=14)
#with open("fixed_windows_results_arr.pickle", "wb") as f:
#    pickle.dump(fixed_windows_results_arr, f)


# In[ ]:


#(hhs_clustered_panel_data).tail()


# In[ ]:


check_slice = hhs_clustered_panel_data[hhs_clustered_panel_data["kmeans_k=3136_labels"]==5]
plt.plot(check_slice["days_from_start"], check_slice["log_rolled_cases"])
plt.show()


# In[ ]:





# In[ ]:




