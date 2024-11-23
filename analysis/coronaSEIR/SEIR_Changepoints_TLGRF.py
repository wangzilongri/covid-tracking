#!/usr/bin/env python
# coding: utf-8

# Standard library imports
import itertools
from itertools import product
#import logging
import joblib
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor


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
filtered_colorado_df = pl.read_csv("filtered_colorado_df.csv").with_columns(pl.col("rolled_cases").shift(7).over("fips").alias("shifted_rolled_cases"))
beginning_of_month_T = pl.read_csv("beginning_of_month_T.csv")


benchmark_schema = {
    'fips': pl.Int64,
    'days_from_start': pl.Int64,
    'intercept_TLGRF': pl.Float64,
    'r_TLGRF': pl.Float64,
    'county': pl.Utf8,  # Equivalent to String in Polars
    'state': pl.Utf8,   # Equivalent to String in Polars
    'date': pl.Date,
    'rolled_cases': pl.Float64,
    'log_rolled_cases': pl.Float64,
    'shifted_log_rolled_cases': pl.Float64,
    'TLGRF_predicted_log_rolled_cases': pl.Float64
}

benchmark_TLGRF_dataset = pl.read_csv("benchmark_TLGRF_dataset.csv", schema=benchmark_schema)
benchmark_colorado_dataset = benchmark_TLGRF_dataset.filter(
        (pl.col("state") == "Colorado") & 
        (pl.col("TLGRF_predicted_log_rolled_cases").is_not_null()) & 
        (pl.col("shifted_log_rolled_cases").is_not_null())
    ).sort(["fips","date"])


#beta_source = pl.read_csv("~/experimental-COVID-tracking/analysis/benchmark_fixed_window/Fixed_windows_all_beta.csv").sort(["fips","days_from_start"])
previous_S = {}
previous_E = {}
previous_I = {}
previous_R = {}
print("Finished loading datasets")

def init_params(fips: int, T_interval: np.ndarray):
    # Obtain earliest t with TLGRF r_tc
    # Then obtain earliest case numbers
    fips_pop =  census.filter(pl.col("FIPS")==fips).select("E_TOTPOP").to_pandas().values[0][0]

    
    fips_df = benchmark_colorado_dataset.filter(pl.col("fips")==fips)
    fips_df = fips_df.filter(pl.col("days_from_start") < T_interval[1])
    fips_df = fips_df.filter(pl.col("days_from_start") >= T_interval[0])

    # Check if empty
    correctness = True

    if fips_df.is_empty():
        tqdm.write("fips={} is empty from {} <= T < {}".format(fips, T_interval[0], T_interval[1]))
        correctness = False
        return {"T": []}, correctness
    else:
        tqdm.write("Initializing fips={} for {} <= T < {}".format(fips, T_interval[0], T_interval[1]))
    
    # Obtain beta_tc
    test_rtc = fips_df.select(['days_from_start','r_TLGRF']).with_columns(
        pl.col("r_TLGRF")
        .fill_null(strategy="backward")  # Backfill initial NaNs
        .fill_null(strategy="forward")  # Forward fill remaining NaNs
    ).to_pandas().values
    test_rolled_cases = fips_df.select(['days_from_start','rolled_cases']).to_pandas().values

    test_shifted_rolled_cases = fips_df.select(['days_from_start','shifted_log_rolled_cases']).to_pandas().values
    test_shifted_rolled_cases[:,1] = np.exp(test_shifted_rolled_cases[:,1])

    
    T = fips_df.select(['days_from_start']).to_pandas().values.astype(int).reshape(1,-1)[0]
    date = fips_df.select(['date']).to_pandas().values.reshape(1,-1)[0]
    
    # Extract the required columns and return as a tuple
    result = {
        "T": T,
        "date": date,
        "r_tc": test_rtc,
        "test_rolled_cases": test_rolled_cases,
        "test_shifted_rolled_cases" :test_shifted_rolled_cases,
        "I_init" : np.exp(fips_df.head(1)['log_rolled_cases'][0]),
        "N_init": fips_pop,
        "S_init": fips_pop - np.exp(fips_df.head(1)['log_rolled_cases'][0])
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
    r_tc: np.ndarray = field(init=False)
    test_rolled_cases: np.ndarray = field(init=False)
    test_shifted_rolled_cases: np.ndarray = field(init=False)

    I_init: float = field(init=False)
    S_init: float = field(init=False)
    N_init: int = field(init=False)
    prev_E: float = None  # New parameter for inheriting the previous value of E
    
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
            self.r_tc = params['r_tc']
            self.test_rolled_cases = params['test_rolled_cases']
            self.test_shifted_rolled_cases = params['test_shifted_rolled_cases']
            
            self.I_init = params['I_init']
            self.S_init = params['S_init']
            self.N_init = params['N_init']


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
            self.r_tc = None
            self.test_rolled_cases = None
            self.test_shifted_rolled_cases = None

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
    N = d_params.N_init  # Population of country
    S, E, I, R = state
    
    #print("Current t={}, state={}".format(t, state))
    
    # Time steps are "continuous", use latest value of beta_t,c
    #print(d_params.beta[d_params.beta[:, 0] <= t])
    beta = d_params.beta[d_params.beta[:, 0] <= t][-1, 1]
    
    sigma = d_params.sigma
    gamma = d_params.gamma

    dS = - beta * S * I / N
    dE = beta * S * I / N - sigma * E
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
    global previous_S
    global previous_E
    global previous_R

    key = (d_params.sigma, d_params.gamma)
    if key not in previous_R:
        # Run in chunks    
        Y0 = [d_params.S_init, 0, d_params.I_init, 0]  # S, E, I, R at initial step
    else:
        Y0 = [d_params.N_init - previous_E[key] - d_params.I_init - previous_R[key], previous_E[key], d_params.I_init, previous_R[key]]  # S, E, I, R
    
    Y_RESULTS = scipy.integrate.solve_ivp(model_seir, t_span=[T[0], T[-1]],
                                          y0=Y0, args=(d_params,),
                                          t_eval=T)

    S, E, I, R = Y_RESULTS.y  # transpose and unpack

    
    results_dict = {"T":T, "S":S, "E":E, "I":I, "R":R}
    results_dict["N"] = d_params.N_init
    results_dict["rolled_cases"] = d_params.test_rolled_cases[:,1]
    results_dict["shifted_rolled_cases"] = d_params.test_shifted_rolled_cases[:,1]
    results_dict["sigma"] = d_params.sigma
    results_dict["gamma"] = d_params.gamma
    
    results_dict["date"] = d_params.date
    
    # Compute R0
    R0 = (d_params.r_tc[:,1]+d_params.gamma)*(d_params.r_tc[:,1]+d_params.sigma)/(d_params.sigma*d_params.gamma)
    results_dict["R0"] = R0
    
    rmse = mean_squared_error(results_dict["rolled_cases"], results_dict["I"])**0.5
    mae = mean_absolute_error(results_dict["rolled_cases"], results_dict["I"])
    
    df = pd.DataFrame(results_dict)

    # Set the 'T' column as the index
    df.set_index('T', inplace=True)


    return df, rmse, mae



def run_model_with_full_predictions(d_params: DiseaseParams):
    """
    Solves the SEIR model for each time step and predicts:
      1. Current day's I, S, E, R values
      2. I, S, E, R values at t + 7 with a fixed beta_t.
      
    :param d_params: DiseaseParams dataclass containing model parameters
    :returns: DataFrame with current day and 7-day future predictions for I, S, E, R
    """
    T = d_params.T  # time-step Array
    global previous_S
    global previous_E
    global previous_I
    global previous_R

    key = (d_params.sigma, d_params.gamma)
    if key not in previous_R:
        # Run in chunks    
        Y0 = [d_params.S_init, 0, d_params.I_init, 0]  # S, E, I, R at initial step
    else:
        #Y0 = [d_params.N_init - previous_E[key] - d_params.I_init - previous_R[key], previous_E[key], d_params.I_init, previous_R[key]]  # S, E, I, R
        Y0 = [d_params.N_init - previous_E[key] - previous_I[key]  - previous_R[key], previous_E[key], previous_I[key], previous_R[key]]  # S, E, I, R

    # Initialize storage for predictions
    current_day_S = []
    current_day_E = []
    current_day_R = []
    current_day_I = []
    
    future_7_day_S = []
    future_7_day_E = []
    future_7_day_R = []
    future_7_day_I = []

    for i, t in enumerate(T):
        tqdm.write(f"(T:{t}, sigma:{d_params.sigma},gamma:{d_params.gamma}): Y0 = {Y0}")
        # Fix beta_t for this time point
        beta_t = max(0,d_params.beta[d_params.beta[:, 0] <= t][-1, 1])

        # Define a wrapper function for the SEIR model with fixed beta
        def fixed_beta_seir(t, state):
            S, E, I, R = state
            N = d_params.N_init
            sigma = d_params.sigma
            gamma = d_params.gamma
            dS = - beta_t * S * I / N
            dE = beta_t * S * I / N - sigma * E
            dI = sigma * E - gamma * I
            dR = gamma * I
            return [dS, dE, dI, dR]

        # Solve the SEIR model for the current day (t)
        t_span_current = [t, t + 1]  # Very small step to get the current prediction
        Y_RESULTS_CURRENT = scipy.integrate.solve_ivp(
            fixed_beta_seir,
            t_span=t_span_current,
            y0=Y0,
            t_eval=[t+1]
        )

        # Extract current day state values
        S_current_day = Y_RESULTS_CURRENT.y[0, -1]
        E_current_day = Y_RESULTS_CURRENT.y[1, -1]
        I_current_day = Y_RESULTS_CURRENT.y[2, -1]
        R_current_day = Y_RESULTS_CURRENT.y[3, -1]

        current_day_S.append(S_current_day)
        current_day_E.append(E_current_day)
        current_day_R.append(R_current_day)
        current_day_I.append(I_current_day)

        # Solve the SEIR model for t -> t + 7
        t_span_future = [t, t + 7]
        Y_RESULTS_FUTURE = scipy.integrate.solve_ivp(
            fixed_beta_seir,
            t_span=t_span_future,
            y0=Y0,
            t_eval=[t + 7]
        )

        # Extract future state values at t + 7
        S_7_days = Y_RESULTS_FUTURE.y[0, -1]
        E_7_days = Y_RESULTS_FUTURE.y[1, -1]
        I_7_days = Y_RESULTS_FUTURE.y[2, -1]
        R_7_days = Y_RESULTS_FUTURE.y[3, -1]

        future_7_day_S.append(S_7_days)
        future_7_day_E.append(E_7_days)
        future_7_day_R.append(R_7_days)
        future_7_day_I.append(I_7_days)

        # Update initial conditions for the next time step
        if i < len(T) - 1:
            Y0 = [S_current_day, E_current_day, I_current_day, R_current_day]

    # Prepare results DataFrame
    results_dict = {
        "T": T,
        "date": d_params.date,
        "S": current_day_S,
        "E": current_day_E,
        "I": current_day_I,
        "R": current_day_R,
        #"future_7_day_S": future_7_day_S,
        #"future_7_day_E": future_7_day_E,
        #"future_7_day_R": future_7_day_R,
        "future_7_day_I": future_7_day_I
    }
    df = pd.DataFrame(results_dict)

    df["N"] = d_params.N_init
    df["rolled_cases"] = d_params.test_rolled_cases[:,1]
    df["shifted_rolled_cases"] = d_params.test_shifted_rolled_cases[:,1]
    
    df["beta"] = d_params.beta[:,1]
    df["sigma"] = d_params.sigma
    df["gamma"] = d_params.gamma
    
    
    # Compute R0
    R0 = (d_params.r_tc[:,1]+d_params.gamma)*(d_params.r_tc[:,1]+d_params.sigma)/(d_params.sigma*d_params.gamma)
    df["R0"] = R0
    
    rmse = mean_squared_error(df["rolled_cases"], df["I"])**0.5
    mae = mean_absolute_error(df["rolled_cases"], df["I"])
    
    # Set the 'T' column as the index
    #df.set_index('T', inplace=True)


    return df, rmse, mae


def simulate(args):
    fips, T_interval, sigma, gamma = args
    global previous_S
    global previous_E
    global previous_I
    global previous_R
    #print(f"Processing: fips={fips}, T_interval={T_interval}, sigma={sigma}, gamma={gamma}")
    disease_params = DiseaseParams(fips=fips, T_interval=T_interval, sigma=sigma, gamma=gamma, prev_E=previous_E)
    if not disease_params.correctness:
        return args, (None, None, None, None)
    #results, rmse, mae = run_model(disease_params)
    results, rmse, mae = run_model_with_full_predictions(disease_params)
    key = (sigma, gamma)

    previous_S[key] = results["S"].iloc[-1]
    previous_E[key] = results["E"].iloc[-1]  # Last E value
    previous_I[key] = results["I"].iloc[-1]
    previous_R[key] = results["R"].iloc[-1]

    tqdm.write(f"Successfully generated results for {fips} at {T_interval}")
    return args, (disease_params, results, rmse, mae)


def process_and_save_results(results_subset):
    best_results = {}
    #print(results_subset)
    # Iterate through the subset results to find the minimum rmse
    for (fips, T_interval, sigma, gamma), (disease_params, result_df, rmse, mae) in results_subset:
        if disease_params is None:
            continue

        key = (fips, tuple(T_interval))
        if key not in best_results or rmse < best_results[key]['rmse']:
            best_results[key] = {
                'sigma': sigma,
                'gamma': gamma,
                'result_df': result_df,
                'rmse': rmse,
                'mae': mae
            }

    # Save the best results to the appropriate directories
    for (fips, T_interval), data in best_results.items():
        # Create the directory if it does not exist
        dir_path = f"./sim_results/{fips}"
        os.makedirs(dir_path, exist_ok=True)

        # Save the result dataframe
        result_file_path = os.path.join(dir_path, f"TLGRF_results_{T_interval[0]}_{T_interval[1]}.csv")
        tqdm.write(f"Saving predictions to {result_file_path}")
        data['result_df'].to_csv(result_file_path, index=False)

        # Save the metadata (fips, T_interval, sigma, gamma, rmse)
        metadata = {
            'fips': fips,
            'T_interval_start': T_interval[0],
            'T_interval_end': T_interval[1],
            'sigma': data['sigma'],
            'gamma': data['gamma'],
            'rmse': data['rmse'],
            'mae': data['mae']
        }
        metadata_file_path = os.path.join(dir_path, f"TLGRF_metadata_{T_interval[0]}_{T_interval[1]}.txt")
        tqdm.write(f"Writing to {metadata_file_path}")
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

def process_fips(fips, T_interval_list, gamma_list, sigma_list):
    try:
        tqdm.write(f"Generating combinations for fips={fips}")
        combinations = list(product([fips], T_interval_list, gamma_list, sigma_list))
        os.makedirs(f"./sim_results/{fips}", exist_ok=True)

        results = []
        for combination in tqdm(combinations, desc=f"Simulating for fips={fips}"):
            args, (disease_params, result_df, rmse, mae, final_E) = simulate(combination)
            results.append((args, (disease_params, result_df, rmse, mae)))

        tqdm.write(f"Saving best rmse for fips={fips}, T_interval combo")
        process_and_save_results(results)
        return fips, True
    except Exception as e:
        tqdm.write(f"Error for fips={fips}: {e}")
        return fips, False

    
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
    gamma_list = np.arange(0.05, 1, 0.05)
    sigma_list = np.arange(0.05, 1, 0.05)
    
    gamma_list = np.arange(0.90, 1.00, 0.01)
    sigma_list = np.arange(0.01, 0.05, 0.01)

    
    # Create directory to store results for each fips
    os.makedirs("./sim_results", exist_ok=True)
    os.makedirs("./dump", exist_ok=True)
    # Return all the results
    num_partitions = multiprocessing.cpu_count()  # Number of partitions (equal to the number of CPU cores)

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
        os.makedirs(f"./sim_results/{fips}", exist_ok=True)

        # Run this per fips
        results = []
        for combination in tqdm(combinations):
            
            results.append(simulate(combination))

        #joblib_file_path = f"./dump/{fips}/results.joblib"
        #joblib.dump(results, joblib_file_path)
        #tqdm.write(f"Saved results for fips={fips} at {joblib_file_path}")
        
        tqdm.write("Saving best rmse for fips={}, T_interval combo".format(fips))
        process_and_save_results(results)
        #run_parallel_processing(results, num_partitions)


