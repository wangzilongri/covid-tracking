#!/usr/bin/env python
# coding: utf-8

# Standard library imports
import itertools
from itertools import product
#import logging
import multiprocessing
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

# Third-party imports
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.widgets  # Cursor
import numpy as np
import pandas as pd
import polars as pl
import scipy.integrate
import scipy.ndimage.interpolation
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Local application imports
#from corona_model import PROJECT_DIR, log_pth
#from corona_model.countryinfo import CountryInfo
#from corona_model.params import DiseaseParams, SimOpts, PlotOpts
#from corona_model.world_data import CovidData



#logger = logging.getLogger(__name__)
#logging.getLogger("matplotlib").setLevel(logging.INFO)

print("Preparing datasets")

changepoints = pl.read_csv("~/experimental-COVID-tracking/case_study/data/CDPHE_TLGRF_historical.csv")
census = pl.read_csv("SVI_2020_US_county.csv")
filtered_colorado_df = pl.read_csv("filtered_colorado_df.csv")
beginning_of_month_T = pl.read_csv("beginning_of_month_T.csv")

print("Finished loading datasets")

def init_params(fips: int, T_interval: np.ndarray):
    # Obtain earliest t with TLGRF r_tc
    # Then obtain earliest case numbers
    fips_pop =  census.filter(pl.col("FIPS")==fips).select("E_TOTPOP").to_pandas().values[0][0]
    
    fips_df = filtered_colorado_df.filter(pl.col("fips")==fips)
    fips_df = fips_df.filter(pl.col("days_from_start") < T_interval[1])
    fips_df = fips_df.filter(pl.col("days_from_start") >= T_interval[0])
    
    # Check if empty
    correctness = True

    if fips_df.is_empty():
        print("fips={} is empty from {} <= T < {}".format(fips, T_interval[0], T_interval[1]))
        correctness = False
        return {"T": []}, correctness
    # Step 1: Sort the data by 'days_from_start'
    test_df = fips_df.sort("days_from_start")
    
    # Obtain beta_tc
    test_rtc = fips_df.select(['days_from_start','r_TLGRF']).to_pandas().values
    test_rolled_cases = fips_df.select(['days_from_start','rolled_cases']).to_pandas().values
    T = fips_df.select(['days_from_start']).to_pandas().values.astype(int).reshape(1,-1)[0]
    date = fips_df.select(['date']).to_pandas().values.reshape(1,-1)[0]
    
    # Extract the required columns and return as a tuple
    result = {
        "T": T,
        "date": date,
        "r_init": fips_df.head(1)['r_TLGRF'][0],
        "r_tc": test_rtc,
        "test_rolled_cases": test_rolled_cases,
        "I_init" :fips_df.head(1)['rolled_cases'][0],
        "S_init": fips_pop
    }
    return result, correctness

@dataclass()
class DiseaseParams:
    """
    Disease parameters. Model is VERY sensitive to these, so they must be picked carefully from
    good sources.
    """
    fips: int
    T_interval: np.ndarray
    gamma: float = 1.0 / (10 + 3) 
    # The rate at which an exposed person becomes infective.
    sigma: float = 1.0 / (5 - 3)

    # New attributes to be initialized via init_params
    date: datetime.date = field(init=False)
    T: np.ndarray = field(init=False)
    r_init: float = field(init=False)
    r_tc: np.ndarray = field(init=False)
    test_rolled_cases: np.ndarray = field(init=False)
    I_init: float = field(init=False)
    S_init: int = field(init=False)
    
    beta: np.ndarray = field(init=False) 
    correctness: bool = field(init=False)
    
    def __post_init__(self):
        # Call init_params to get the initialization values
        params, correctness = init_params(self.fips, self.T_interval)
        self.correctness = correctness            
        if self.correctness:
            # Set the attributes from the dictionary
            self.T = params['T']
            self.date = params['date']
            self.r_init = params['r_init']
            self.r_tc = params['r_tc']
            self.test_rolled_cases = params['test_rolled_cases']
            self.I_init = params['I_init']
            self.S_init = params['S_init']


            # Initialize test_beta with the same number of rows as r_tc and 2 columns
            test_beta = np.zeros_like(self.r_tc)

            # Update test_beta array: first column from r_tc, second column calculation
            test_beta[:, 0] = self.r_tc[:, 0].astype(int)
            test_beta[:, 1] = ((self.r_tc[:, 1] * 2 + (self.sigma + self.gamma))**2 - (self.sigma - self.gamma)**2) / (4 * self.sigma)

            # Convert the list of tuples to a tuple of tuples
            self.beta = test_beta
        else:
            # Set the attributes from the dictionary
            self.T = None
            self.date = None
            self.r_init = None
            self.r_tc = None
            self.test_rolled_cases = None
            self.I_init = None
            self.S_init = None
            self.beta = None



def model_seir(t: float, state: Iterable[np.ndarray], d_params: DiseaseParams) -> Tuple[float, float, float, float, float]:
    """
    Definition of SEIR model
    :param t: Time-step (days), dependant variable of ODEs
    :param state: Vector of ODE State variables [S, E, I, R]
    :param d_params: DiseaseParams dataclass from params, or your own/modified version
    :param s_opts: SimOpts dataclass from params, or your own/modified version
    :returns: 4-element tuple of change in each of the state variables
    """
    N = d_params.S_init  # Population of country
    S, E, I, R = state
    
    #print("Current t={}, state={}".format(t, state))
    
    # Time steps are "continuous", use latest value of beta_t,c
    beta = d_params.beta[d_params.beta[:, 0] <= t][-1, 1]
    
    sigma = d_params.sigma
    gamma = d_params.gamma

    dS = - beta * S * I / N
    dE = beta * S * I / N  - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I

    return dS, dE, dI, dR

def run_model(d_params: DiseaseParams) -> Tuple[float, float, float, float]:
    """
    Solves the ODE model and returns results.
    :param d_params: d_params: DiseaseParams dataclass from params, or your own/modified version
    :param s_opts: s_opts: SimOpts dataclass from params, or your own/modified version
    :returns: 5-element Tuple of arrays of results
    """
    T = d_params.T  # time-step Array
    
    # Run in chunks    
    Y0 = [d_params.S_init - d_params.I_init, 0, d_params.I_init, 0]  # S, E, I, R at initial step


    
    Y_RESULTS = scipy.integrate.solve_ivp(model_seir, t_span=[T[0], T[-1]],
                                          y0=Y0, args=(d_params,),
                                          t_eval=T)

    S, E, I, R = Y_RESULTS.y  # transpose and unpack

    
    results_dict = {"T":T, "S":S, "E":E, "I":I, "R":R}
    results_dict["rolled_cases"] = d_params.test_rolled_cases[:,1]
    results_dict["date"] = d_params.date
    
    # Compute R0
    R0 = (d_params.r_tc[:,1]+d_params.gamma)*(d_params.r_tc[:,1]+d_params.sigma)/(d_params.sigma*d_params.gamma)
    results_dict["R0"] = R0
    
    rmse = mean_squared_error(results_dict["rolled_cases"], results_dict["I"])**0.5
    
    df = pd.DataFrame(results_dict)

    # Set the 'T' column as the index
    df.set_index('T', inplace=True)

    return df, rmse


def simulate(args):
    fips, T_interval, sigma, gamma = args
    #print(f"Processing: fips={fips}, T_interval={T_interval}, sigma={sigma}, gamma={gamma}")
    disease_params = DiseaseParams(fips=fips, T_interval=T_interval, sigma=sigma, gamma=gamma)
    if not disease_params.correctness:
        return args, (None, None, None)
    results, rmse = run_model(disease_params)

    return args, (disease_params, results, rmse)


def process_and_save_results(results_subset):
    best_results = {}

    # Iterate through the subset results to find the minimum rmse
    for (fips, T_interval, sigma, gamma), (disease_params, result_df, rmse) in results_subset:
        if disease_params is None:
            continue

        key = (fips, tuple(T_interval))
        if key not in best_results or rmse < best_results[key]['rmse']:
            best_results[key] = {
                'sigma': sigma,
                'gamma': gamma,
                'result_df': result_df,
                'rmse': rmse
            }

    # Save the best results to the appropriate directories
    for (fips, T_interval), data in best_results.items():
        # Create the directory if it does not exist
        dir_path = f"./sim_results/{fips}"
        os.makedirs(dir_path, exist_ok=True)

        # Save the result dataframe
        result_file_path = os.path.join(dir_path, f"results_{T_interval[0]}_{T_interval[1]}.csv")
        data['result_df'].to_csv(result_file_path, index=False)

        # Save the metadata (fips, T_interval, sigma, gamma, rmse)
        metadata = {
            'fips': fips,
            'T_interval_start': T_interval[0],
            'T_interval_end': T_interval[1],
            'sigma': data['sigma'],
            'gamma': data['gamma'],
            'rmse': data['rmse']
        }
        metadata_file_path = os.path.join(dir_path, f"metadata_{T_interval[0]}_{T_interval[1]}.txt")
        with open(metadata_file_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

def partition_results(results, num_partitions):
    """Partition results into chunks for parallel processing."""
    #print("Partitioning len(results)={}, num_partitions={}".format(len(results), num_partitions))
    avg_chunk_size = max(len(results) // num_partitions,1)
    return [results[i:i + avg_chunk_size] for i in range(0, len(results), avg_chunk_size)]

def run_parallel_processing(results, num_partitions):
        """Process results in parallel."""
        partitions = partition_results(results, num_partitions)

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            pool.map(process_and_save_results, partitions)

    
    
if __name__ == "__main__":
    """
    changepoints = pl.read_csv("~/experimental-COVID-tracking/case_study/data/CDPHE_TLGRF_historical.csv")
    census = pl.read_csv("SVI_2020_US_county.csv")
    fips_list = (colorado_rtc.select("fips").unique().to_pandas().values.reshape(1,-1))[0]
    filtered_colorado_df = pl.read_csv("filtered_colorado_df.csv")
    beginning_of_month_T = pl.read_csv("beginning_of_month_T.csv")

    """
    T_points_list = beginning_of_month_T.to_pandas().values[:,1]

    # Generates list of stuff to parallelize over
    fips_list = sorted((filtered_colorado_df.select("fips").unique().to_pandas().values.reshape(1,-1))[0])
    T_interval_list = np.array([T_points_list[i:i+2] for i in range(0, len(T_points_list) - 1)])
    gamma_list = np.arange(0.05, 1.0, 0.05)
    sigma_list = np.arange(0.05, 1.0, 0.05)
    
    # Create directory to store results for each fips
    os.makedirs("./sim_results", exist_ok=True)
    
    # 0 to 63
    fips_list_start, fips_list_end = 0, 64
    try:
        fips_list_start, fips_list_end = int(sys.argv[1]), int(sys.argv[2])
    except:
        print("Need arguments, from 0 to 64 for start and end")
        
    # Generate list of arguments:
    for fips in tqdm(fips_list[fips_list_start:fips_list_end]):
        print("Generating combinations for fips={}".format(fips))
        combinations = list(product([fips], T_interval_list, gamma_list, sigma_list))

        # Run this per fips
        try:
            results = []
            for combination in tqdm(combinations):
                results.append(simulate(combination))

            # Return all the results
            num_partitions = multiprocessing.cpu_count()  # Number of partitions (equal to the number of CPU cores)

            print("Saving best rmse for fips={}, T_interval combo".format(fips))
            process_and_save_results(results)
            #run_parallel_processing(results, num_partitions)
        except:
            print("Error for fips={}".format(fips))


