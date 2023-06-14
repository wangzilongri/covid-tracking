import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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


from multiprocessing import Pool, cpu_count

import dask
import dask.dataframe as dd
from dask.distributed import Client
#client = Client(n_workers=20, memory_limit="10GB", interface='lo')
from concurrent.futures import ThreadPoolExecutor

import dask_ml.cluster as dask_cluster

from pprint import pprint
import os

pd.set_option('display.max_columns', None)


merged_TLGRF_results = dd.read_csv("../data/output/merged_TLGRF_results_df.csv", assume_missing=True).compute()

cols_to_keep = ["fips", "county_x", "state_x", "date.x", "days_from_start", "tau.hat", "log_rolled_cases", "shifted_log_rolled_cases", "predicted.grf.future.last"]
kept_merged_TLGRF_results = merged_TLGRF_results[cols_to_keep]
kept_merged_TLGRF_results["date.x"] = pd.to_datetime(kept_merged_TLGRF_results["date.x"])

kept_merged_TLGRF_results = kept_merged_TLGRF_results.rename(columns={"date.x":"date", "county_x":"county", "state_x": "state"})

kept_merged_TLGRF_results.to_csv("benchmark_TLGRF_dataset.csv", index=False)
